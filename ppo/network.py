import math
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensor_annotations import axes
import tensor_annotations.tensorflow as ttf
from tensorflow import keras
import env
from typing import NewType, TypeAlias

ActionAxis: TypeAlias = NewType('Actions', axes.Axis)

# Hyperparameters
BOARD_CONV_FILTERS = 8
HIDDEN_LAYER_SIZE = 50

LR = 1e-5  # Lower lr stabilises training greatly
GAMMA = 0.99

# PPO2
ACTOR_PPO_LOSS_CLIPPING=0.2
PPO_TRAINING_EPO = 5
# H stands for entropy here
H_TARGET = 0.1

class PPOAgent:
    def __init__(self, width:int, height:int):
        self.board_width = width
        self.board_height = height
        self.actor = self.__build_actor()
        self.critic = self.__build_critic()
        self._entropy_weight:float = np.log(width)


    # Private
    def __build_actor(self):
        # 2D Matrix of int8 inputs
        feature_board = keras.layers.Input((self.board_height, self.board_width), dtype=np.int8)
        # 2D Matrix of float32
        float32_board = keras.layers.Lambda(lambda x: tf.cast(x, dtype=np.float32))(feature_board)
        # convert into matrix with channels
        board_with_channels = tf.keras.layers.Reshape(
            target_shape=(self.board_height, self.board_width, 1),
            input_shape=(self.board_height, self.board_width)
        )(float32_board)

        # convolve the board so that the network can focus on key features
        convolved_board = keras.layers.Conv2D(BOARD_CONV_FILTERS, (4, 4), activation="relu")(board_with_channels)

        # flatten the convolved board and concatenate it with other data
        # this will be used as input for the dense hidden layers
        hidden_layer_in = keras.layers.Flatten()(convolved_board)
        # now 2 layers of hidden board size
        hidden_layer0_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_in)
        hidden_layer1_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer0_out)
        hidden_layer2_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer1_out)

        action_probs = keras.layers.Dense(self.board_width, activation='softmax')(hidden_layer2_out)

        # oldpolicy probs is the vector of predictions made by the old actor model
        oldpolicy_probs = keras.layers.Input(shape=self.board_width)
        # is a 1-hot vector cooresponding to the chosen_action (which may be different than the argmax since we add gumbel noise)
        action_chosen = keras.layers.Input(shape=self.board_width)

        # advantage captures how better an action is compared to the others at a given state
        # https://theaisummer.com/Actor_critics/
        advantage = keras.layers.Input(shape=(1,))

        # Needed for entropy regularization
        entropy_weight = keras.layers.Input(shape=(1,))

        model = keras.Model(
            inputs=[
                # ppo features
                advantage,
                oldpolicy_probs,
                action_chosen,
                entropy_weight,
                # real features
                feature_board,
            ],
            outputs=[action_probs]
        )


        def actor_ppo_loss(
            newpolicy_probs: ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
            oldpolicy_probs: ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
            action_chosen: ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
            advantage: ttf.Tensor1[ttf.float32,axes.Batch],
            entropy_weight: ttf.Tensor1[ttf.float32, axes.Batch],
        ):
            # Calculates the likelyhood ratio
            # This is used to penalize excessive differences between the original and new policies
            def calculate_likelyhood_ratio(
                pi_new:ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
                pi_old:ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
                acts:ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch]
            ) -> ttf.Tensor1[ttf.float32, axes.Batch]:
                pi_new_prob:ttf.Tensor1[ttf.float32, axes.Batch] = tf.reduce_sum(tf.multiply(pi_new, acts), axis=1)
                pi_old_prob:ttf.Tensor1[ttf.float32, axes.Batch] = tf.reduce_sum(tf.multiply(pi_old, acts), axis=1)
                return pi_new_prob / pi_old_prob

            # likelyhood ratio w 
            w = calculate_likelyhood_ratio(newpolicy_probs, oldpolicy_probs, action_chosen)

            # PPO2 Loss Clipping
            ppo2loss = tf.minimum(
                w*advantage,
                tf.clip_by_value(w, 1-ACTOR_PPO_LOSS_CLIPPING, 1+ACTOR_PPO_LOSS_CLIPPING)*advantage
            )

            # Dual Loss (Unsure what the advantage of this is???)
            dual_loss = tf.where(tf.less(advantage, 0.), tf.maximum(ppo2loss, 3. * advantage), ppo2loss)

            # Calculate Entropy (how uncertain the prediction is)
            entropy = -tf.reduce_sum(tf.multiply(newpolicy_probs, tf.math.log(newpolicy_probs)), axis=1, keepdims=True)

            # actor loss
            return -tf.reduce_sum(dual_loss) - entropy_weight * entropy

        model.add_loss(actor_ppo_loss(
          newpolicy_probs=action_probs,
          oldpolicy_probs=oldpolicy_probs,
          action_chosen=action_chosen,
          advantage=advantage,
          entropy_weight=entropy_weight
        ))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LR),
        )

        return model

    # Private
    # The critic attempts to learn the advantage
    def __build_critic(self):
        # 2D Matrix of int8 inputs
        feature_board = keras.layers.Input((self.board_height, self.board_width), dtype=np.int8)
        # 2D Matrix of float32
        float32_board = keras.layers.Lambda(lambda x: tf.cast(x, dtype=np.float32))(feature_board)
        # convert into matrix with channels
        board_with_channels = tf.keras.layers.Reshape(
            target_shape=(self.board_height, self.board_width, 1),
            input_shape=(self.board_height, self.board_width)
        )(float32_board)

        # convolve the board so that the network can focus on key features
        convolved_board = keras.layers.Conv2D(BOARD_CONV_FILTERS, (4, 4), activation="relu")(board_with_channels)

        # flatten the convolved board and concatenate it with other data
        # this will be used as input for the dense hidden layers
        hidden_layer_in = keras.layers.Flatten()(convolved_board)
        # now 2 layers of hidden board size
        hidden_layer0_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_in)
        hidden_layer1_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer0_out)
        hidden_layer2_out = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer1_out)

        value = keras.layers.Dense(1, activation='linear')(hidden_layer2_out)

        model = keras.Model(
            inputs=[feature_board],
            outputs=[value]
        )

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss='mse')

        return model

    def save(self, actor_path:str, critic_path:str):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load(self, actor_path:str, critic_path:str):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def get_network_params(self):
        return [
          self.actor.get_weights(),
          self.critic.get_weights(),
        ]

    def set_network_params(self, weights):
        self.actor.set_weights(weights[0])
        self.critic.set_weights(weights[1])

    def predict_batch(
        self,
        observation_batch: list[env.Observation],
    ) -> npt.NDArray[np.float32]:
        batch_len = len(observation_batch)

        # Convert state batch into correct format
        board_batched = np.zeros((batch_len, self.board_height, self.board_width))
        for i, (o,) in enumerate(observation_batch):
            board_batched[i] = o

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage_batched = np.zeros((batch_len, 1))
        oldpolicy_probs_batched = np.zeros((batch_len, self.board_width))
        chosen_action_batched = np.zeros((batch_len, self.board_width))
        entropy_weight_batched = np.zeros((batch_len, 1))

        p = self.actor(
            [
                # Dummy
                advantage_batched,
                oldpolicy_probs_batched,
                chosen_action_batched,
                entropy_weight_batched,
                # Real
                board_batched,
            ],
        )
        return p

    def critic_batch(
        self,
        observation_batch: list[env.Observation]
    ) -> npt.NDArray[np.float32]:
        batch_len = len(observation_batch)
        # Convert state batch into correct format
        board_batched = np.zeros((batch_len, self.board_height, self.board_width))
        for i, (o,) in enumerate(observation_batch):
            board_batched[i] = o
        
        return self.critic([board_batched])

    def train(
        self,
        observation_batch: list[env.Observation],
        action_batch: list[env.Action],
        advantage_batch:list[float],
        old_prediction_batch: list[npt.NDArray[np.float32]],
        base_step:int
    ):
        batch_len = len(observation_batch)
        assert batch_len == len(action_batch)
        assert batch_len == len(advantage_batch) 
        assert batch_len == len(old_prediction_batch)

        # Convert state batch into correct format
        board_batched = np.zeros((batch_len, self.board_height, self.board_width))
        for i, (o,) in enumerate(observation_batch):
            board_batched[i] = o

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage_batched = np.reshape(advantage_batch, (batch_len, 1))
        oldpolicy_probs_batched = np.reshape(old_prediction_batch, (batch_len, self.board_width))
        # create 1 hot vector
        chosen_action_batched = np.zeros((batch_len, self.board_width))
        chosen_action_batched[np.arange(0, batch_len), action_batch] = 1

        entropy_weight = np.full((batch_len, 1), self._entropy_weight);

        class PrintLoss(keras.callbacks.Callback):
            def __init__(self, base_step:int, name:str):
                self.base_step = base_step
                self.name = name
            def on_epoch_end(self, epoch, logs):
                tf.summary.scalar(self.name, logs['loss'], step=self.base_step+epoch)

        # Train Actor
        self.actor.fit(
            [
                # Required to compute loss
                advantage_batched,
                oldpolicy_probs_batched,
                chosen_action_batched,
                entropy_weight,
                # Real values
                board_batched,
            ],
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'actor_loss')]
        )

        # Train Critic
        self.critic.fit(
            board_batched,
            advantage_batched,
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'critic_loss')]
        )

        p_batch = np.clip(oldpolicy_probs_batched, ACTOR_PPO_LOSS_CLIPPING, 1. - ACTOR_PPO_LOSS_CLIPPING)
        H = np.mean(np.sum(-np.log(p_batch) * p_batch, axis=1))
        g = H - H_TARGET
        self._entropy_weight -= LR * float(g) * 0.1

        return PPO_TRAINING_EPO

    def get_entropy_weight(self) -> float:
        return self._entropy_weight

    # computes advantage
    def compute_advantage(
        self,
        state_batch: list[env.Observation],
        reward_batch: list[env.Reward],
        terminal: bool
    ) -> list[float]:
        assert len(state_batch) == len(reward_batch)

        batch_len = len(state_batch)
        R_batch = np.zeros(len(reward_batch))

        if terminal:
            R_batch[-1] = 0  # terminal state
        else:
            # calculate the advantage at the end
            v_batch = self.critic_batch([state_batch[-1]])
            R_batch[-1] = v_batch[0]

        # Use GAMMA to decay the advantage 
        for t in reversed(range(batch_len- 1)):
            R_batch[t] = reward_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)

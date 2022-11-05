import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensor_annotations import axes
import tensor_annotations.tensorflow as ttf
import env
import random
import math
from typing import NewType, TypeAlias

LR = 0.0001
BOARD_CONV_FILTERS = 8
HIDDEN_LAYER_SIZE = 50
GAMMA = 0.99

# used for epsilon greedy selection
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Deep Q network

ActionAxis: TypeAlias = NewType('Actions', axes.Axis)


def build_network(dims: tuple[int, int]):
    # 2D Matrix of int8 inputs
    observation_input = keras.layers.Input(shape=dims, dtype=ttf.int8)

    # convolve the board so that the network can focus on key features
    convolved_board = keras.layers.Conv2D(
        BOARD_CONV_FILTERS,
        (4, 4), activation="relu")(observation_input)

    # flatten the convolved board and concatenate it with other data
    # this will be used as input for the dense hidden layers
    hidden_layer_in = keras.layers.Concatenate()(
        keras.layers.Flatten()(convolved_board),
    )
    # now 2 layers of hidden board size
    hidden_layer0_out = keras.layers.Dense(
        shape=HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_in)
    hidden_layer1_out = keras.layers.Dense(
        shape=HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer0_out)
    hidden_layer2_out = keras.layers.Dense(
        shape=HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer1_out)

    # yield out the expected reward for each given action
    # 1D Array of rewards, one for each action
    q_policy_pred_output = keras.layers.Dense(
        shape=dims[1], activation='linear')(hidden_layer2_out)

    # this is the rewards predicted for each action at time t+1 in trajectory tau_j
    # predicted by target network
    # Q_target(s_{t+1}, a) for all valid a
    q_target_nextstate_pred_input = keras.layers.Input(
        shape=dims[1], dtype=ttf.float32)

    # this is the actual reward we got during the transition to the next state
    actual_env_reward_input = keras.layers.Input(shape=(1,), dtype=np.int8)

    # this is the actual action we selected
    selected_action_input = keras.layers.Input(shape=(1,), dtype=np.int8)

    model = keras.Model(
        inputs=[
            observation_input,
            # used to calculate loss
            q_target_nextstate_pred_input,
            actual_env_reward_input,
            selected_action_input
        ],
        outputs=[q_policy_pred_output]
    )

    huberloss = keras.loss.Huber()

    def dqn_loss(
        # for each index $j$ in batch:
        # this is the rewards predicted for each action at time t in trajectory tau_j
        # predicted by policy network
        # Q_policy(s_t, a) for all valid a
        q_policy_pred: ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
        # for each index $j$ in batch:
        # this is the rewards predicted for each action at time t+1 in trajectory tau_j
        # predicted by target network
        # Q_target(s_{t+1}, a) for all valid a
        q_target_nextstate_pred: ttf.Tensor2[ttf.float32, ActionAxis, axes.Batch],
        # for each index $j$ in batch:
        # this is the actual reward we got during the transition to the next state
        actual_env_reward: ttf.Tensor1[ttf.float32, axes.Batch],
        # for each index $j$ in batch:
        # this is the actual action we selected
        selected_action: ttf.Tensor1[ttf.int8, axes.Batch],
    ):
        # get the Q value for s, a from the q_policy_pred
        # this is done by selecting the q value for the thing we already chose
        q_s_a_t: ttf.Tensor1[ttf.float32, axes.Batch] = tf.gather(
            q_policy_pred, selected_action, 1)

        # compute V(s_{t+1}) by taking the max reward predicted by the target network
        v_s_t1: ttf.Tensor1[ttf.float32, axes.Batch] = tf.math.maximum(
            q_target_nextstate_pred, 1)

        # this is what should have been the Q value for s and a
        # it adds the real reward that we got at time t and a time discounted future reward
        expected_q_s_a_t = (v_s_t1 * GAMMA) + actual_env_reward

        # do huber loss between the real and expected q values
        return huberloss(q_s_a_t, expected_q_s_a_t)

    model.add_loss(dqn_loss(
        q_policy_pred=q_policy_pred_output,
        q_target_nextstate_pred=q_target_nextstate_pred_input,
        actual_env_reward=actual_env_reward_input,
        selected_action=selected_action_input
    ))

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=LR), loss='mse')

    return model


class DeepQNetwork:
    def __init__(self, dims: tuple[int, int]):
        self.dims = dims
        self.network = build_network(dims)

    def save(self, network_path: str):
        self.network.save_weights(network_path)

    def load(self, network_path: str):
        self.network.load_weights(network_path)

    def predict_action(self,
        # the step we're on right now (used for determining how random)
        base_step: int,
        # the observation of the board
        observation: env.Observation 
    ) -> env.Action:
        # generate the q values for each action
        observation_input = observation

        # there are many inputs to the network that would be used
        # if we needed to calculate loss, but we don't at this point
        # therefore we will replace them with zeros
        q_target_nextstate_pred_input = np.zeros(self.dims[1], dtype=np.float32)
        actual_env_reward_input = np.array(0.0)
        selected_action_input = np.array(0.0)

        q_values:ttf.Tensor1[ttf.float32, ActionAxis] = self.network(
            [
                np.array([observation_input]),
                # these are not used for prediction
                np.array([q_target_nextstate_pred_input]),
                np.array([actual_env_reward_input]),
                np.array([selected_action_input])
            ],
            training=False
        )[0]


        # alpha is the how far we are along the decay
        alpha = math.exp(-base_step / EPS_DECAY)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * alpha

        # with probability epsilon we select a random action
        sample = random.random()

        if sample > eps_threshold:
            return np.argmax(np.array(q_values))
        else:
            return np.int8(random.randrange(self.dims[1]))

    def train(
        self,
        state_batch: list[env.Observation],
        action_batch: list[env.Action],
        old_prediction_batch: list[env.Action],
        base_step:int
    ):
        batch_len = len(state_batch)
        assert batch_len == len(action_batch)
        assert batch_len == len(advantage_batch) 
        assert batch_len == len(old_prediction_batch)

        # Convert state batch into correct format
        historical_network_throughput = np.zeros((len(state_batch), self.network_history_len)) 
        historical_chunk_download_time = np.zeros((len(state_batch), self.network_history_len)) 
        available_video_bitrates = np.zeros((len(state_batch), self.available_video_bitrates_count))
        buffer_level = np.zeros((len(state_batch), 1))
        remaining_chunk_count  = np.zeros((len(state_batch), 1))
        last_chunk_bitrate  = np.zeros((len(state_batch), 1))

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage = np.reshape(advantage_batch, (batch_len, 1))
        oldpolicy_probs= np.reshape(old_prediction_batch, (batch_len, self.available_video_bitrates_count))
        chosen_action = np.reshape(action_batch, (batch_len, self.available_video_bitrates_count))

        entropy_weight = np.full((batch_len, 1), self._entropy_weight);

        for (i, (hnt, hcdt, avb, bl, rcc, lcb)) in enumerate(state_batch):
            historical_network_throughput[i] = hnt
            historical_chunk_download_time[i] = hcdt
            available_video_bitrates[i] = avb
            buffer_level[i] = bl
            remaining_chunk_count[i] = rcc
            last_chunk_bitrate[i] = lcb

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
                advantage,
                oldpolicy_probs,
                chosen_action,
                entropy_weight,
                # Real values
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'loss_actor')]
        )

        # Train Critic
        self.critic.fit(
            [
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
            advantage,
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'critic_loss')]
        )

        p_batch = np.clip(oldpolicy_probs, ACTOR_PPO_LOSS_CLIPPING, 1. - ACTOR_PPO_LOSS_CLIPPING)
        H = np.mean(np.sum(-np.log(p_batch) * p_batch, axis=1))
        g = H - H_TARGET
        self._entropy_weight -= LR * g * 0.1

        return PPO_TRAINING_EPO


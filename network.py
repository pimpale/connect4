import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensor_annotations import axes
import tensor_annotations.tensorflow as ttf
import env
from typing import NewType,TypeAlias

LR = 0.0001
BOARD_CONV_FILTERS = 8
HIDDEN_LAYER_SIZE = 50
GAMMA = 0.99

# Deep Q network

Actions:TypeAlias = NewType('Actions', axes.Axis)

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
    q_target_nextstate_pred_input = keras.layers.Input(shape=dims[1], dtype=ttf.float32)

    # this is the actual reward we got during the transition to the next state
    actual_env_reward_input = keras.layers.Input(shape=(1,), dtype=ttf.float32)

    # this is the actual action we selected
    selected_action_input = keras.layers.Input(shape=(1,), dtype=ttf.float32)

    model = keras.Model(
        inputs=[
            observation_input,
            # used to calculate loss
            q_target_nextstate_pred_input ,
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
        q_policy_pred:ttf.Tensor2[ttf.float32, Actions, axes.Batch],
        # for each index $j$ in batch:
        # this is the rewards predicted for each action at time t+1 in trajectory tau_j
        # predicted by target network
        # Q_target(s_{t+1}, a) for all valid a
        q_target_nextstate_pred:ttf.Tensor2[ttf.float32, Actions, axes.Batch],
        # for each index $j$ in batch:
        # this is the actual reward we got during the transition to the next state
        actual_env_reward:ttf.Tensor1[ttf.float32, axes.Batch],
        # for each index $j$ in batch:
        # this is the actual action we selected
        selected_action:ttf.Tensor1[ttf.int8, axes.Batch],
    ):
        # get the Q value for s, a from the q_policy_pred
        # this is done by selecting the q value for the thing we already chose
        q_s_a_t:ttf.Tensor1[ttf.float32, axes.Batch] = tf.gather(q_policy_pred, selected_action, 1)

        # compute V(s_{t+1}) by taking the max reward predicted by the target network
        v_s_t1:ttf.Tensor1[ttf.float32, axes.Batch] = tf.math.maximum(q_target_nextstate_pred, 1)

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

    def select_action(self, observation:

    def predict_batch(
        self,
        state_batch: list[env.Observation],
    ):

        batch_len = len(state_batch)

        # Convert state batch into correct format
        board_state_batch = np.zeros(
            (batch_len, self.dims[0], self.dims[1]), dtype=np.int8)

        for (i, board_state) in enumerate(state_batch):
            historical_network_throughput[i] = hnt
            historical_chunk_download_time[i] = hcdt
            available_video_bitrates[i] = avb
            buffer_level[i] = bl
            remaining_chunk_count[i] = rcc
            last_chunk_bitrate[i] = lcb

        p = self.actor(
            [
                # Dummy
                advantage,
                oldpolicy_probs,
                chosen_action,
                entropy_weight,
                # Real
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
        )
        return p

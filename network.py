import numpy as np
from tensorflow import keras

import env

LR=0.0001
BOARD_CONV_FILTERS = 8
HIDDEN_LAYER_SIZE = 200


def build_network(dims: tuple[int, int]):
    # 2D Matrix of int8 inputs
    board_state_input = keras.layers.Input(shape=dims, dtype=np.int8)
    # convolve the board so that the network can focus on key features
    convolved_board = keras.layers.Conv2D(
        8, (4, 4), activation="relu")(board_state_input)
    # flatten the convolved board
    convolved_flat = keras.layers.Flatten()(convolved_board)
    # now 2 layers of hidden board size
    hidden_layer0_out = keras.layers.Dense(
        HIDDEN_LAYER_SIZE, activation='relu')(convolved_flat)
    hidden_layer1_out = keras.layers.Dense(
        HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer0_out)

    # yield out probabilities for the output, each one corresponding to a different location
    action_probs = keras.layers.Dense(dims[1], activation='softmax')(hidden_layer1_out)

    model = keras.Model(
        inputs=[
            board_state_input,
        ],
        outputs=[action_probs]
    )

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=LR), loss='mse')

    return model


class DeepQNetwork:
    def __init__(self, dims: tuple[int, int]):
        self.dims = dims
        self.network = build_network(dims)
    
    def save(self, network_path:str):
        self.network.save_weights(network_path)

    def load(self, network_path:str):
        self.network.load_weights(network_path)

    def predict_batch(
        self,
        state_batch: list[env.Observation],
    ):

        batch_len = len(state_batch)

        # Convert state batch into correct format
        board_state_batch = np.zeros((batch_len, self.dims[0], self.dims[1]), dtype=np.int8) 

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

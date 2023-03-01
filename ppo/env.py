import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import Optional, TypeAlias

@dataclass
class State:
    """State of the game"""
    board: npt.NDArray[np.int8]

@dataclass
class Observation:
    """Observation by a single player of the game"""
    board:npt.NDArray[np.int8]

# Column to place it in
Action:TypeAlias = np.int8

# Reward for the agent
Reward:TypeAlias = np.float32

# Value of an observation for an agent
Value:TypeAlias = np.float32

# Advantage of a particular action for an agent
Advantage:TypeAlias = np.float32

PLAYER1 = np.int8(1)
PLAYER2 = np.int8(2)

def print_obs(o:Observation):
    for row in reversed(o.board):
        # We print '#' for our item, and 'O' for the opponent
        for x in row:
            c = ' '
            if x == 1:
                c = '#'
            elif x == 2:
                c = 'O'
            print(c, end=" ")
        print()
    print()

def initial_state(dims:tuple[int, int]) -> State:
    return State(np.zeros(dims, dtype=np.int8))

def state_to_observation(state: State, actor: np.int8) -> Observation:
    s = state.board
    o = s.copy()
    o[s == actor] = 1
    o[s != actor] = 2
    o[s == 0] = 0
    return Observation(o)


horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


def is_winner(state:State, actor:np.int8) -> bool:
    board = state.board
    for kernel in detection_kernels:
        if (convolve2d(board == actor, kernel, mode="valid") == 4).any():
            return True
    return False

# returns if the board is completely filled
def drawn(state:State) -> bool:
    return 0 not in state.board

# return the reward for the actor
def state_to_reward(s: State, actor: np.int8) -> Reward:
    opponent = PLAYER2 if actor == PLAYER1 else PLAYER1
    
    if is_winner(s, actor):
        return np.float32(1.0)
    elif is_winner(s, opponent):
        return np.float32(-1.0)
    else:
        return np.float32(0.0)

class Env():
    def __init__(
        self,
        dims:tuple[int,int]
    ):
        self._game_over = False
        self.state: State = initial_state(dims)

    def reset(self) -> None:
        self.state = initial_state(self.state.board.shape)

    def observe(self, actor: np.int8) -> Observation:
        return state_to_observation(self.state, actor)

    def game_over(self) -> bool:
        return self._game_over

    def legal_mask(self) -> npt.NDArray[np.bool8]:
        return self.state.board[-1] == 0

    def step(self, a: Action, actor: np.int8) -> tuple[Reward, Observation]:
        for row in self.state.board:
            if row[a] == 0:
                row[a] = actor
                break

        r = state_to_reward(self.state, actor)
        o = state_to_observation(self.state, actor)

        if r != 0:
            self._game_over = True
        elif drawn(self.state):
            self._game_over = True

        return (r, o)

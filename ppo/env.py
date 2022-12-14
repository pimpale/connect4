import numpy as np
import numpy.typing as npt;
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


def print_obs(o:Observation):
    for row in reversed(o.board):
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

def winner(state:State) -> Optional[np.int8]:
    s = state.board
    ysize, xsize = s.shape
    # check horizontal
    for y in range(0, ysize):
        for x in range(0, xsize-3):
            actor = s[y][x]
            if actor != 0 and s[y][x+1] == actor and s[y][x+2] == actor and s[y][x+3] == actor:
                return actor
    # check vertical
    for x in range(0, xsize):
        for y in range(0, ysize-3):
            actor = s[y][x]
            if actor != 0 and s[y+1][x] == actor and s[y+2][x] == actor and s[y+3][x] == actor:
                return actor
    # check diagonals 1 way
    for y in range(0, ysize-3):
        for x in range(0, xsize-3):
            actor = s[y][x]
            if actor != 0 and s[y+1][x+1] == actor and s[y+2][x+2] == actor and s[y+3][x+3] == actor:
                return actor
    # check diagonals other way
    for y in range(0, ysize-3):
        for x in range(3, xsize):
            actor = s[y][x]
            if actor != 0 and s[y+1][x-1] == actor and s[y+2][x-2] == actor and s[y+3][x-3] == actor:
                return actor
    # finally return None if no winner found
    return None

# returns if the board is completely filled
def drawn(state:State) -> bool:
    return 0 not in state.board

# return the reward for the actor
def state_to_reward(s: State, actor: np.int8) -> Reward:
    w = winner(s)
    if w == actor:
        return np.float32(1.0)
    elif w is not None:
        return np.float32(-1.0)
    else:
        return np.float32(0.0)
    
class Env():
    def __init__(
        self,
        dims:tuple[int,int]
    ):
        self.state: State = initial_state(dims)

    def reset(self) -> None:
        self.state = initial_state(self.state.board.shape)

    def observe(self, actor: np.int8) -> Observation:
        return state_to_observation(self.state, actor)

    def game_over(self) -> bool:
        if winner(self.state) is not None:
            return True
        else:
            return drawn(self.state)

    def step(self, a: Action, actor: np.int8) -> tuple[Reward, Observation]:
        for row in self.state.board:
            if row[a] == 0:
                row[a] = actor
                break

        r = state_to_reward(self.state, actor)
        o = state_to_observation(self.state, actor)

        return (r, o)

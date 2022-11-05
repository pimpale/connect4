import numpy as np
import numpy.typing as npt;
from typing import Optional, TypeAlias

# State of the board
State:TypeAlias = npt.NDArray[np.int8]

# Observation of the board
Observation:TypeAlias = npt.NDArray[np.int8]

# Column to place it in
Action:TypeAlias = np.int8

# Reward for the agent
Reward:TypeAlias = np.float32


def initial_state(dims:tuple[int, int]) -> State:
    x= np.zeros(dims, dtype=np.int8)
    x[1] = 2
    return np.zeros(dims, dtype=np.int8)


def state_to_observation(s: State, actor: np.int8) -> Observation:
    o: Observation = s.copy()
    o[s == actor] = 1
    o[s != actor] = 2
    o[s == 0] = 0
    return o

def winner(s:State) -> Optional[np.int8]:
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

# return the reward for the actor
def state_to_reward(s: State, actor: np.int8) -> Reward:
    if winner(s) == actor:
        return np.float32(1.0)
    else:
        return np.float32(0.0)
    
class Env():
    def __init__(
        self,
        dims:tuple[int,int]
    ):
        self.dims = dims
        self.state: State = initial_state(dims)

    def reset(self) -> None:
        self.state = initial_state(self.dims)

    def observe(self, actor: np.int8) -> Observation:
        return state_to_observation(self.state, actor)

    def game_over(self) -> bool:
        return winner(self.state) is not None

    def step(self, a: Action, actor: np.int8) -> tuple[Reward, Observation]:
        full: bool = True
        for i in range(self.state.shape[1]):
            if self.state[i][a] == 0:
                self.state[i][a] = actor
                full = False
                break

        # if the column is full then this action is illegal
        if full:
            raise ValueError("column is full!")

        r = state_to_reward(self.state, actor)
        o = state_to_observation(self.state, actor)

        return (r, o)

    def render(self) -> None:
        for row in reversed(self.state):
            for x in row:
                print(x, end=" ")
            print()
        print()

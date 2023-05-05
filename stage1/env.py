import numpy as np
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import Any, TypeAlias, Literal

@dataclass
class State:
    """State of the game"""
    board: np.ndarray[Any, np.dtype[np.int8]]

@dataclass
class Observation:
    """Observation by a single player of the game"""
    board: np.ndarray[Any, np.dtype[np.int8]]


# Column to place it in
Action:TypeAlias = np.int8

# Reward for the agent
Reward:TypeAlias = np.float32

# Value of an observation for an agent
Value:TypeAlias = np.float32

# Advantage of a particular action for an agent
Advantage:TypeAlias = np.float32

Player:TypeAlias = np.int8

PLAYER1:Player = np.int8(1)
PLAYER2:Player = np.int8(2)

def opponent(actor:Player) -> Player:
    return PLAYER2 if actor == PLAYER1 else PLAYER1

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


class Env():
    def __init__(
        self,
        dims:tuple[int,int]
    ):
        self._game_over = False
        self._winner = None
        self._moves = []
        self.state: State = initial_state(dims)

    def reset(self) -> None:
        self._game_over = False
        self._winner = None
        self._moves = []
        self.state = initial_state(self.state.board.shape)

    def observe(self, actor: np.int8) -> Observation:
        return state_to_observation(self.state, actor)

    def game_over(self) -> bool:
        return self._game_over

    def winner(self) -> Player | None:
        return self._winner

    def dims(self) -> tuple[int, int]:
        return self.state.board.shape

    def legal_mask(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.state.board[-1] == 0

    def legal_actions(self) -> list[Action]:
        return [Action(i) for i in range(self.dims()[1]) if self.legal_mask()[i]]

    def step(self, a: Action, actor: Player) -> Reward:
        ### YOUR CODE HERE: ###
        # 1. We assume the move is legal. Modify the game board to reflect the move.
        # 1.5 Add the move to self._moves.
        # 2. Check if the game is over. If so, set self._game_over and self._winner.
        # 3. Return the reward for the agent.
        pass
    
    def undo(self):
        if len(self._moves) == 0:
            return
        self.state.board[self._moves.pop()] = 0
        self._winner = None
        self._game_over = False

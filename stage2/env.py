import numpy as np
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import Any, Self, TypeAlias, Literal

BOARD_XSIZE = 7
BOARD_YSIZE = 6

# Column to place it in
Action: TypeAlias = np.int8
Player: TypeAlias = np.int8

PLAYER1: Player = np.int8(1)
PLAYER2: Player = np.int8(2)


@dataclass
class State:
    """State of the game"""

    board: np.ndarray[Any, np.dtype[np.int8]]
    current_player: Player

    def legal_mask(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.board[-1] == 0

    def legal_actions(self) -> np.ndarray[Any, np.dtype[np.int8]]:
        # return the indices of the legal actions
        return np.where(self.legal_mask())[0]

    def copy(self) -> Self:
        return State(self.board.copy(), self.current_player)


def opponent(actor: Player) -> Player:
    return PLAYER2 if actor == PLAYER1 else PLAYER1


def print_state(s: State):
    board = s.board
    current_player = s.current_player
    opponent_player = opponent(current_player)
    for y in reversed(range(BOARD_YSIZE)):
        # We print '#' for current player, and 'O' for the opponent
        for x in range(BOARD_XSIZE):
            c = " "
            if board[y, x] == current_player:
                c = "#"
            elif board[y, x] == opponent_player:
                c = "O"
            print(c, end=" ")
        print()
    print()


horizontal_kernel = np.array([[1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


def is_winner(state: State, actor: Player) -> bool:
    board = state.board
    for kernel in detection_kernels:
        if (convolve2d(board == actor, kernel, mode="valid") == 4).any():
            return True
    return False


# returns if the board is completely filled
def drawn(state: State) -> bool:
    return 0 not in state.board


# return the reward for the actor
def state_to_reward(s: State, player: Player) -> float:
    if is_winner(s, player):
        return np.float32(1.0)
    elif is_winner(s, opponent(player)):
        return np.float32(-1.0)
    else:
        return np.float32(0.0)


class Env:
    def __init__(self):
        self._game_over = False
        self._winner = None
        # contains the location of the last placed square
        self._moves = []
        self.state: State = State(
            board=np.zeros((BOARD_YSIZE, BOARD_XSIZE), dtype=np.int8),
            current_player=PLAYER1,
        )

    def reset(self) -> None:
        self._game_over = False
        self._winner = None
        self._moves = []
        self.state = State(
            board=np.zeros((BOARD_YSIZE, BOARD_XSIZE), dtype=np.int8),
            current_player=PLAYER1,
        )

    def game_over(self) -> bool:
        return self._game_over

    def winner(self) -> Player | None:
        return self._winner

    def dims(self) -> tuple[int, int]:
        return self.state.board.shape

    def step(self, a: Action) -> float:
        for i, row in enumerate(self.state.board):
            if row[a] == 0:
                self._moves.append((i, a))
                row[a] = self.state.current_player
                break

        # set next player to go
        self.state.current_player = opponent(self.state.current_player)

        r = state_to_reward(self.state, self.state.current_player)

        if r != 0:
            self._game_over = True
            self._winner = self.state.current_player
        elif drawn(self.state):
            self._game_over = True

        return r

    def undo(self):
        if len(self._moves) == 0:
            return
        self.state.board[self._moves.pop()] = 0
        self.state.current_player = opponent(self.state.current_player)
        self._winner = None
        self._game_over = False

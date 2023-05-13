from abc import ABC, abstractmethod
import numpy as np
import math
import scipy.stats
from scipy.signal import convolve2d


import env


class Player(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class RandomPlayer(Player):
    def __init__(self, player: env.Player) -> None:
        self.player = player

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        p = legal_mask / np.sum(legal_mask)
        chosen_action = env.Action(np.random.choice(len(p), p=p))
        reward = e.step(chosen_action, self.player)
        return (obs, chosen_action, reward)

    def name(self) -> str:
        return "random"


class HumanPlayer(Player):
    def __init__(self, player: env.Player) -> None:
        self.player = player

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        env.print_obs(obs)
        print("0 1 2 3 4 5 6")
        print("legal mask:", legal_mask, flush=True)
        chosen_action = np.int8(input("Choose action: "))
        reward = e.step(chosen_action, self.player)

        return (obs, chosen_action, reward)

    def name(self) -> str:
        return "human"


# this heuristic just counts the number of 4-in-a-rows each player has
# returns a number between 0 and 1
def heuristic(e: env.Env) -> float:
    player1_valid = e.observe(env.PLAYER1).board != env.PLAYER2
    player2_valid = e.observe(env.PLAYER2).board != env.PLAYER1

    player1_score = 0
    player2_score = 0

    for kernel in env.detection_kernels:
        player1_score += np.sum(convolve2d(player1_valid, kernel, mode="valid") == 4)
        player2_score += np.sum(convolve2d(player2_valid, kernel, mode="valid") == 4)

    return np.tanh(player1_score - player2_score)


# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(
    e: env.Env, depth: int, alpha: float, beta: float, player: env.Player
) -> tuple[float, env.Action | None]:
    # if a winner has been decided
    winner = e.winner()
    if winner is not None:
        reward = 1 if winner == env.PLAYER1 else -1
        return reward, None

    legal_actions = e.legal_actions()
    # if we have no legal moves (draw)
    if len(legal_actions) == 0:
        return 0, None

    # if we've reached the depth limit
    if depth == 0:
        return heuristic(e), None

    if player == env.PLAYER1:
        best_score = -math.inf
        best_action = None
        for action in legal_actions:
            reward = e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER2)
            e.undo()
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_action
    else:
        best_score = math.inf
        best_action = None
        for action in legal_actions:
            reward = e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER1)
            e.undo()
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best_score, best_action


class MinimaxPlayer(Player):
    def __init__(self, player: env.Player, depth: int, randomness: float) -> None:
        self.player = player
        self.depth = depth
        self.randomness = randomness

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        # introduce some randomness
        if np.random.random() < self.randomness:
            return RandomPlayer(self.player).play(e)

        obs = e.observe(self.player)
        _, chosen_action = minimax(e, self.depth, -math.inf, math.inf, self.player)
        if chosen_action is not None:
            reward = e.step(chosen_action, self.player)
            return (obs, chosen_action, reward)
        else:
            return RandomPlayer(self.player).play(e)

    def name(self) -> str:
        return f"minimax(depth={self.depth},randomness={self.randomness})"

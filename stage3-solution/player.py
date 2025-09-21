from abc import ABC, abstractmethod
import numpy as np
import math
import torch
from scipy.signal import convolve2d


import env
import network


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

def heuristic(e: env.Env) -> float:
    player1_placed = e.observe(env.PLAYER1).board == env.PLAYER1
    player2_placed = e.observe(env.PLAYER2).board == env.PLAYER1

    player1_score = 0
    player2_score = 0

    for kernel in env.detection_kernels:
        player1_convolved = convolve2d(player1_placed, kernel, mode="valid")
        player2_convolved = convolve2d(player2_placed, kernel, mode="valid")
        player1_score += np.sum(player1_convolved == 3)
        player2_score += np.sum(player2_convolved == 3)

    return np.tanh(player1_score - player2_score)



# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(
    e: env.Env, depth: int, alpha: float, beta: float, player: env.Player
) -> tuple[float, env.Action]:
    winner = e.winner()
    if winner is not None:
        reward = math.inf if winner == env.PLAYER1 else -math.inf
        return reward, env.Action(0)
    if depth == 0:
        return heuristic(e), env.Action(0)

    if player == env.PLAYER1:
        best_score = -math.inf
        best_action = env.Action(0)
        for action in e.legal_actions():
            e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER2)
            e.undo()
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_action
    else:
        best_score = math.inf
        best_action = env.Action(0)
        for action in e.legal_actions():
            e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER1)
            e.undo()
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, best_score)
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
        reward = e.step(chosen_action, self.player)

        return (obs, chosen_action, reward)

    def name(self) -> str:
        return f"minimax(depth={self.depth},randomness={self.randomness})"

class ActorPlayer(Player):
    def __init__(
        self,
        actor: network.Actor,
        name: str,
        player: env.Player,
    ) -> None:
        self.actor = actor
        self.player = player
        self.name = name

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(self.player)

        device = network.deviceof(self.actor)

        action_probs = (
            self.actor.forward(network.obs_batch_to_tensor([obs], device))[0]
            .detach()
            .cpu()
            .numpy()
        )

        legal_mask = e.legal_mask()

        raw_p = action_probs * legal_mask
        p = raw_p / np.sum(raw_p)

        chosen_action = env.Action(np.random.choice(len(p), p=p))
        reward = e.step(chosen_action, self.player)

        return (obs, chosen_action, reward)

    def name(self) -> str:
        return f"actor_{self.name}"

def ActorCheckpointPlayer(ActorPlayer):
    def __init__(self, checkpoint_path: str, player: env.Player, board_xsize: int, board_ysize: int) -> None:
        # load the actor from the checkpoint
        actor = network.Actor(board_xsize, board_ysize)
        actor.load_state_dict(torch.load(checkpoint_path))
        super().__init__(actor, checkpoint_path, player)

    def name(self) -> str:
        return f"actor_checkpoint_{self.checkpoint_path}"
from abc import ABC, abstractmethod
import random
from typing import Self
import numpy as np
import math
from pydantic import BaseModel
import torch
from scipy.signal import convolve2d


import env
import network


class Policy(BaseModel, ABC):
    @abstractmethod
    def __call__(self, env: env.Env) -> env.Action: ...

    @classmethod
    def fmt_config(cls, model_dict: dict) -> str:
        # print like this: {policy_class.__name__}(key=value, key=value, ...)
        config_str = ", ".join([f"{key}={value}" for key, value in model_dict.items()])
        return f"{cls.__name__}({config_str})"


class RandomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> env.Action:
        legal_mask = s.legal_mask()
        p = legal_mask / np.sum(legal_mask)
        return env.Action(np.random.choice(len(p), p=p))

class HumanPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> env.Action:
        env.print_state(s)
        print("0 1 2 3 4 5 6")
        legal_mask = s.legal_mask()
        print("legal mask:", legal_mask)
        chosen_action = np.int8(input("Choose action: "))
        return env.Action(chosen_action)

def heuristic(s: env.State) -> float:
    self_placed = s.board == s.current_player
    opponent_placed = s.board == env.opponent(s.current_player)

    self_score = 0
    opponent_score = 0

    for kernel in env.detection_kernels:
        self_convolved = convolve2d(self_placed, kernel, mode="valid")
        opponent_convolved = convolve2d(opponent_placed, kernel, mode="valid")
        self_score += np.sum(self_convolved == 3)
        opponent_score += np.sum(opponent_convolved == 3)

    return np.tanh(self_score - opponent_score)


# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(
    e: env.Env, depth: int, alpha: float, beta: float
) -> tuple[float, env.Action]:
    winner = e.winner()
    if winner is not None:
        reward = math.inf if winner == env.PLAYER1 else -math.inf
        return reward, env.Action(0)
    # if the game is drawn return 0
    if e.game_over():
        return 0, env.Action(0)
    if depth == 0:
        return heuristic(e.state), env.Action(0)

    if e.state.current_player == env.PLAYER1:
        best_score = -math.inf
        legal_actions = e.state.legal_actions()
        np.random.shuffle(legal_actions)
        best_action = legal_actions[0]
        for action in legal_actions:
            e.step(action)
            score, _ = minimax(e, depth - 1, alpha, beta)
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
        legal_actions = e.state.legal_actions()
        np.random.shuffle(legal_actions)
        best_action = legal_actions[0]
        for action in legal_actions:
            e.step(action)
            score, _ = minimax(e, depth - 1, alpha, beta)
            e.undo()
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_action


class MinimaxPolicy(Policy):
    depth: int
    randomness: float

    def __init__(self, depth: int, randomness: float):
        super().__init__(depth=depth, randomness=randomness)

    def __call__(self, s: env.State) -> env.Action:
        # introduce some randomness
        if np.random.random() < self.randomness:
            return RandomPolicy()(s)

        # create a new env and set the state
        e = env.Env()
        e.state = s

        _, chosen_action = minimax(e, self.depth, -math.inf, math.inf)

        return chosen_action

class NNCheckpointPolicy(Policy):
    _actor: network.Actor
    checkpoint_path: str

    def __init__(
        self,
        checkpoint_path: str,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path)
        # load the actor from the checkpoint
        actor = network.Actor(env.BOARD_XSIZE, env.BOARD_YSIZE)
        actor.load_state_dict(torch.load(checkpoint_path))
        actor.eval()
        self._actor = actor

    def __call__(self, s: env.State) -> env.Action:
        device = network.deviceof(self._actor)

        action_probs = (
            self._actor.forward(network.state_batch_to_tensor([s], device))[0]
            .detach()
            .cpu()
            .numpy()
        )

        legal_mask = s.legal_mask()

        raw_p = action_probs * legal_mask
        p = raw_p / np.sum(raw_p)

        chosen_action = env.Action(np.random.choice(len(p), p=p))

        return chosen_action

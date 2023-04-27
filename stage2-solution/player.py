from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats
from scipy.signal import convolve2d
import math

import env

class Player(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

class RandomPlayer(Player):
    def __init__(self, player:env.Player) -> None:
        self.player = player
    
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        action_prob = scipy.special.softmax(np.random.random(size=len(legal_mask)))
        chosen_action: env.Action = np.argmax(action_prob*legal_mask)
        reward = e.step(chosen_action, self.player)
    
        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return "random"

class HumanPlayer(Player):
    def __init__(self, player:env.Player) -> None:
        self.player = player
    
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        env.print_obs(obs)
        print('0 1 2 3 4 5 6')
        print("legal mask:", legal_mask)
        chosen_action = np.int8(input("Choose action: "))
        reward = e.step(chosen_action, self.player)
    
        return (
            obs,
            legal_mask,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return "human"
    



# this heuristic just counts the number of 4-in-a-rows each player has
def heuristic(e:env.Env) -> float:
    player1_valid = e.observe(env.PLAYER1).board != env.PLAYER2
    player2_valid = e.observe(env.PLAYER2).board != env.PLAYER1

    player1_score = 0
    player2_score = 0

    for kernel in env.detection_kernels:
        player1_score += np.sum(convolve2d(player1_valid, kernel, mode="valid") == 4)
        player2_score += np.sum(convolve2d(player2_valid, kernel, mode="valid") == 4)

    return player1_score - player2_score


# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(e:env.Env, depth:int, alpha:float, beta:float, player:env.Player) -> tuple[float, env.Action]:
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
            score,_ = minimax(e, depth-1, alpha, beta, env.PLAYER2)
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
            score,_ = minimax(e, depth-1, alpha, beta, env.PLAYER1)
            e.undo()
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_action

class MinimaxPlayer(Player):
    def __init__(self, player:env.Player, depth:int, randomness:float) -> None:
        self.player = player
        self.depth = depth
        self.randomness = randomness
    
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        # introduce some randomness
        if np.random.random() < self.randomness:
            return RandomPlayer(self.player).play(e)
        
        obs = e.observe(self.player)
        _,chosen_action = minimax(e, self.depth, -math.inf, math.inf, self.player)
        action_prob = np.eye(e.dims()[1])[chosen_action]
        reward = e.step(chosen_action, self.player)
    
        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return f"minimax(depth={self.depth},randomness={self.randomness})"
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
    

# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(e:env.Env, depth:int, alpha:float, beta:float, player:env.Player) -> tuple[float, env.Action]:
    pass

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
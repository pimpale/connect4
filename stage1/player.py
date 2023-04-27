from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats


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
        print('0 1 2 3 4 5 6 7')
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
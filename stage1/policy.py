from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats
from pydantic import BaseModel


import env

class Player(ABC, BaseModel):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def play(self, player:env.Player, e:env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        pass

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
    
    def play(self, player:env.Player, e:env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
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

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
    
    def play(self, player:env.Player, e:env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(player)
        legal_mask = e.legal_mask()
        env.print_obs(obs)
        print('0 1 2 3 4 5 6')
        print("legal mask:", legal_mask)
        action_prob = np.un
        chosen_action = np.int8(input("Choose action: "))
        reward = e.step(chosen_action, player)
    
        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )
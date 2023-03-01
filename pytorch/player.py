from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats

import env
import network

class Player(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ActorPlayer(Player):
    def __init__(self, actor:network.Actor, epoch:int, player:np.int8) -> None:
        self.actor = actor
        self.player = player
        self.epoch = epoch

    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(self.player)

        device = network.deviceof(self.actor)

        action_probs = self.actor.forward(network.obs_to_tensor(obs, device))[0].to("cpu").detach().numpy()
                
        action_entropy = scipy.stats.entropy(action_probs)
        if action_entropy < 0.001:
            raise ValueError("Entropy is too low!")
                    
        if np.isnan(action_probs).any():
            raise ValueError("NaN found!")
    
        legal_mask = e.legal_mask() 
    
        action_logprobs = np.log(action_probs)
    
        # apply noise to probs
        noise = 0.1*np.random.gumbel(size=len(action_logprobs))
        adjusted_action_probs = scipy.special.softmax(action_logprobs + noise) 
    
        legal_mask = e.legal_mask() 
    
        chosen_action: env.Action = np.argmax(adjusted_action_probs*legal_mask)
        reward,_ = e.step(chosen_action, self.player)
    
        return (
            obs,
            action_probs,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return f"actor_ckpt_{self.epoch}"

class RandomPlayer(Player):
    def __init__(self, player:np.int8) -> None:
        self.player = player
    
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        action_prob = scipy.special.softmax(np.random.random(size=len(legal_mask)))
        chosen_action: env.Action = np.argmax(action_prob*legal_mask)
        reward,_ = e.step(chosen_action, self.player)
    
        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return "random"
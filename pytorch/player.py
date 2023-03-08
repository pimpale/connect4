from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats
from scipy.signal import convolve2d


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
    def __init__(self, actor:network.Actor, critic:network.Critic, epoch:int, player:env.Actor) -> None:
        self.actor = actor
        self.critic = critic
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
    def __init__(self, player:env.Actor) -> None:
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


# use a convolve2d heuristic to evaluate the board
def heuristic(e:env.Env, player:env.Actor) -> env.Reward:
    # get the opponent
    opponent = env.opponent(player)

    # the board as seen by the player
    player_obs = e.observe(player).board == player
    opponent_obs = e.observe(opponent).board == opponent

    player_score = 0
    opponent_score = 0

    for kernel in env.detection_kernels:
        player_score += np.sum(convolve2d(player_obs, kernel, mode="valid"))
        opponent_score += np.sum(convolve2d(opponent_obs, kernel, mode="valid"))

    score_diff = player_score - opponent_score 
    # logit transform
    return env.Reward(np.exp(score_diff)/(1+np.exp(score_diff)))


# use the minimax algorithm to find the best move, searching up to depth
def minimax(e:env.Env, depth:int, player:env.Actor) -> tuple[env.Reward, env.Action]:
    if depth == 0:
        return (heuristic(e, player), env.Action(0))

    legal_mask = e.legal_mask()
    legal_actions = np.where(legal_mask == 1)[0]

    if len(legal_actions) == 0:
        return (env.Reward(0),env.Action(0))

    best_action = legal_actions[0]
    best_reward:env.Reward = env.Reward(-np.inf)
    for action in legal_actions:
        e.step(action, player)
        reward,_ = minimax(e, depth-1, env.opponent(player))
        e.undo()
        if reward > best_reward:
            best_reward = reward
            best_action = action

    return (best_reward, best_action)

class MinimaxPlayer(Player):
    def __init__(self, player:env.Actor, depth:int) -> None:
        self.player = player
        self.depth = depth
    
    def play(self, e:env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        # introduce some randomness
        if np.random.random() > 0.5:
            return RandomPlayer(self.player).play(e)
        
        obs = e.observe(self.player)
        _,chosen_action = minimax(e, self.depth, self.player)
        action_prob = np.eye(e.dims()[1])[chosen_action]
        reward,_ = e.step(chosen_action, self.player)
    
        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )
    
    def name(self) -> str:
        return "minimax"
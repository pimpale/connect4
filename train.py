import numpy as np
import tensorflow as tf
from collections import deque
import env
from typing import TypeAlias
import random

# consists of an observation, an action we took on it, our reward from the observation, and the resultant observation
Transition:TypeAlias = tuple[env.Observation, env.Action, env.Reward, env.Observation]

class ReplayMemory:
    def __init__(self, capacity:int):
        self.memory:deque[Transition] = deque([],maxlen=capacity)

    def push(self, t:Transition):
        """Save a transition"""
        self.memory.append(t)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



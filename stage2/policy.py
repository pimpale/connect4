from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats
from scipy.signal import convolve2d
import math
from pydantic import BaseModel

import env

class Policy(ABC, BaseModel):
    @abstractmethod
    def __call__(self, e: env.Env) -> env.Action:
        ...

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.model_fields_set)
        return f"{self.__class__.__name__}({args})"

class RandomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, e: env.Env) -> env.Action:
        legal_mask = e.legal_mask()
        action_prob = scipy.special.softmax(np.random.random(size=len(legal_mask)))
        chosen_action: env.Action = np.argmax(action_prob * legal_mask)
        return chosen_action

class HumanPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, e: env.Env) -> env.Action:
        # Display the current board state
        # Note: We'll need to determine which player perspective to show
        # For now, display the raw board
        print("Current board:")
        for row in reversed(e.state.board):
            for x in row:
                c = ' '
                if x == 1:
                    c = '#'
                elif x == 2:
                    c = 'O'
                print(c, end=" ")
            print()
        legal_mask = e.legal_mask()
        print('0 1 2 3 4 5 6')
        print("legal mask:", legal_mask)
        chosen_action = env.Action(np.int8(input("Choose action: ")))
        return chosen_action
    

# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(e:env.Env, depth:int, alpha:float, beta:float, player:env.Player) -> tuple[float, env.Action]:
    # TODO: Insert your code here
    pass

class MinimaxPolicy(Policy):
    depth: int
    randomness: float
    
    def __init__(self, depth: int, randomness: float) -> None:
        super().__init__(depth=depth, randomness=randomness)
    
    def __call__(self, e: env.Env) -> env.Action:
        # introduce some randomness
        if np.random.random() < self.randomness:
            return RandomPolicy()(e)
        
        # For minimax, we need to know which player is making the move
        # Assuming current_player is available in the state
        player = e.state.current_player if hasattr(e.state, 'current_player') else env.PLAYER1
        _, chosen_action = minimax(e, self.depth, -math.inf, math.inf, player)
        return chosen_action
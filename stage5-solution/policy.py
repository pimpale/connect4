from abc import ABC, abstractmethod
from typing import Self, Optional, Dict, List
import numpy as np
import math
import random
from pydantic import BaseModel
from scipy.signal import convolve2d


import env


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


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(
        self, 
        state: env.State, 
        parent: Optional[Self] = None, 
        action: Optional[env.Action] = None,
        player: env.Player = env.PLAYER1
    ) -> None:
        self.state: env.State = state
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[env.Action] = action  # Action that led to this node
        self.player: env.Player = player  # Player who will make a move from this state
        
        self.visits: int = 0
        self.wins: float = 0.0  # Win score from perspective of PLAYER1
        self.children: Dict[env.Action, MCTSNode] = {}
        self.untried_actions: List[env.Action] = self._get_legal_actions()
        
    def _get_legal_actions(self) -> List[env.Action]:
        """Get all legal actions from the current state"""
        # Check which columns are not full
        legal_actions: List[env.Action] = []
        for col in range(self.state.board.shape[1]):
            if self.state.board[-1, col] == 0:  # Top row is not occupied
                legal_actions.append(env.Action(col))
        return legal_actions
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state"""
        # Check for winner
        if env.is_winner(self.state, env.PLAYER1) or env.is_winner(self.state, env.PLAYER2):
            return True
        # Check for draw
        return env.drawn(self.state)
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been expanded"""
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param: float = 1.4142) -> Self:
        """Select the best child using UCB1 formula"""
        choices_weights: List[float] = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')
            else:
                # UCB1 formula
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
                
                # Adjust for the current player's perspective
                if self.player == env.PLAYER2:
                    exploitation = -exploitation
                    
                weight = exploitation + exploration
            choices_weights.append(weight)
        
        return list(self.children.values())[int(np.argmax(choices_weights))]
    
    def expand(self) -> Self:
        """Expand the tree by creating a new child node"""
        action: env.Action = self.untried_actions.pop()
        
        # Create a copy of the state and apply the action
        new_state = env.State(self.state.board.copy(), self.state.current_player)
        
        # Find the first empty row in the chosen column and place the piece
        for row_idx in range(new_state.board.shape[0]):
            if new_state.board[row_idx, action] == 0:
                new_state.board[row_idx, action] = self.player
                break
        
        # Create the child node with the opposite player
        child = MCTSNode(
            state=new_state,
            parent=self,
            action=action,
            player=env.opponent(self.player)
        )
        self.children[action] = child
        return child
    
    def update(self, result: float) -> None:
        """Update node statistics after a simulation"""
        self.visits += 1
        self.wins += result


class MCTSPolicy(Policy):
    """Monte Carlo Tree Search policy"""
    
    simulations: int
    c_param: float
    randomness: float
    
    def __init__(
        self, 
        simulations: int = 1000,
        c_param: float = 1.4142,
        randomness: float = 0.0
    ) -> None:
        """
        Initialize MCTS Policy
        
        Args:
            simulations: Number of simulations to run per move
            c_param: Exploration parameter for UCB1 (higher = more exploration)
            randomness: Probability of making a random move (0.0 = always use MCTS)
        """
        super().__init__(simulations=simulations, c_param=c_param, randomness=randomness)
    
    def _simulate(self, node: MCTSNode) -> float:
        """Run a random simulation from the given node to a terminal state"""
        # Create a temporary environment copy for simulation
        temp_state = env.State(node.state.board.copy(), node.state.current_player)
        current_player = node.player
        
        # Play random moves until the game ends
        while not env.is_winner(temp_state, env.PLAYER1) and \
              not env.is_winner(temp_state, env.PLAYER2) and \
              not env.drawn(temp_state):
            
            # Get legal actions
            legal_actions: List[env.Action] = []
            for col in range(temp_state.board.shape[1]):
                if temp_state.board[-1, col] == 0:
                    legal_actions.append(env.Action(col))
            
            if not legal_actions:
                break
            
            # Choose a random action
            action = random.choice(legal_actions)
            
            # Apply the action
            for row_idx in range(temp_state.board.shape[0]):
                if temp_state.board[row_idx, action] == 0:
                    temp_state.board[row_idx, action] = current_player
                    break
            
            # Switch player
            current_player = env.opponent(current_player)
        
        # Return the result from PLAYER1's perspective
        if env.is_winner(temp_state, env.PLAYER1):
            return 1.0
        elif env.is_winner(temp_state, env.PLAYER2):
            return -1.0
        else:
            return 0.0  # Draw
    
    def _mcts_search(self, root: MCTSNode) -> env.Action:
        """Run MCTS to find the best action"""
        simulations_run = 0
        
        while simulations_run < self.simulations:
            node = root
            
            # Selection: traverse the tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_param)
            
            # Expansion: add a new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation: run a random playout
            result = self._simulate(node)
            
            # Backpropagation: update statistics
            while node is not None:
                node.update(result)
                node = node.parent
            
            simulations_run += 1
        
        # Choose the action with the highest visit count (most robust choice)
        best_action: Optional[env.Action] = None
        best_visits: int = -1
        
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        return best_action if best_action is not None else env.Action(0)
    
    def __call__(self, s: env.State) -> env.Action:
        """Make a move using MCTS"""
        # Introduce randomness if specified
        if np.random.random() < self.randomness:
            return RandomPolicy()(s)
        
        # Determine the current player based on the board state
        # Count the number of pieces to determine whose turn it is
        num_player1 = np.sum(s.board == env.PLAYER1)
        num_player2 = np.sum(s.board == env.PLAYER2)
        current_player = env.PLAYER1 if num_player1 == num_player2 else env.PLAYER2
        
        # Create root node from current state
        root = MCTSNode(
            state=s.copy(),
            parent=None,
            action=None,
            player=current_player
        )
        
        # If there are legal actions, run MCTS
        if root.untried_actions or root.children:
            chosen_action = self._mcts_search(root)
            return chosen_action
        
        # Fallback to random policy if no action found
        return RandomPolicy()(s)

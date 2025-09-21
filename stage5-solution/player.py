from abc import ABC, abstractmethod
import numpy as np
import math
from scipy.signal import convolve2d
from typing import Optional, Dict, List, Self
import random


import env


class Player(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class RandomPlayer(Player):
    def __init__(self, player: env.Player) -> None:
        self.player = player

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        p = legal_mask / np.sum(legal_mask)
        chosen_action = env.Action(np.random.choice(len(p), p=p))
        reward = e.step(chosen_action, self.player)
        return (obs, chosen_action, reward)

    def name(self) -> str:
        return "random"


class HumanPlayer(Player):
    def __init__(self, player: env.Player) -> None:
        self.player = player

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        obs = e.observe(self.player)
        legal_mask = e.legal_mask()
        env.print_obs(obs)
        print("0 1 2 3 4 5 6")
        print("legal mask:", legal_mask, flush=True)
        chosen_action = np.int8(input("Choose action: "))
        reward = e.step(chosen_action, self.player)

        return (obs, chosen_action, reward)

    def name(self) -> str:
        return "human"


# this heuristic just counts the number of 4-in-a-rows each player has
# returns a number between 0 and 1
def heuristic(e: env.Env) -> float:
    player1_placed = e.observe(env.PLAYER1).board == env.PLAYER1
    player2_placed = e.observe(env.PLAYER2).board == env.PLAYER1

    player1_score = 0
    player2_score = 0

    for kernel in env.detection_kernels:
        player1_convolved = convolve2d(player1_placed, kernel, mode="valid")
        player2_convolved = convolve2d(player2_placed, kernel, mode="valid")

        player1_score += 0.2*np.sum(player1_convolved == 2)
        player2_score += 0.2*np.sum(player2_convolved == 2)
        player1_score += np.sum(player1_convolved == 3)
        player2_score += np.sum(player2_convolved == 3)

    return np.tanh(player1_score - player2_score)


# use the minimax algorithm (with alpha beta pruning) to find the best move, searching up to depth
def minimax(
    e: env.Env, depth: int, alpha: float, beta: float, player: env.Player
) -> tuple[float, env.Action | None]:
    # if a winner has been decided
    winner = e.winner()
    if winner is not None:
        reward = 1 if winner == env.PLAYER1 else -1
        return reward, None

    legal_actions = e.legal_actions()
    # if we have no legal moves (draw)
    if len(legal_actions) == 0:
        return 0, None

    # if we've reached the depth limit
    if depth == 0:
        return heuristic(e), None

    if player == env.PLAYER1:
        best_score = -math.inf
        best_action = None
        for action in legal_actions:
            reward = e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER2)
            e.undo()
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_action
    else:
        best_score = math.inf
        best_action = None
        for action in legal_actions:
            reward = e.step(action, player)
            score, _ = minimax(e, depth - 1, alpha, beta, env.PLAYER1)
            e.undo()
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best_score, best_action


class MinimaxPlayer(Player):
    def __init__(self, player: env.Player, depth: int, randomness: float) -> None:
        self.player = player
        self.depth = depth
        self.randomness = randomness

    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        # introduce some randomness
        if np.random.random() < self.randomness:
            return RandomPlayer(self.player).play(e)

        obs = e.observe(self.player)
        best_score, chosen_action = minimax(e, self.depth, -math.inf, math.inf, self.player)
        print(best_score)
        if chosen_action is not None:
            reward = e.step(chosen_action, self.player)
            return (obs, chosen_action, reward)
        else:
            return RandomPlayer(self.player).play(e)

    def name(self) -> str:
        return f"minimax(depth={self.depth},randomness={self.randomness})"


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree with strong typing"""
    
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
        new_state = env.State(self.state.board.copy())
        
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


class MCTSPlayer(Player):
    """Monte Carlo Tree Search player with strong typing"""
    
    def __init__(
        self, 
        player: env.Player, 
        simulations: int = 1000,
        c_param: float = 1.4142,
        randomness: float = 0.0
    ) -> None:
        """
        Initialize MCTS Player
        
        Args:
            player: Which player this is (PLAYER1 or PLAYER2)
            simulations: Number of simulations to run per move
            c_param: Exploration parameter for UCB1 (higher = more exploration)
            randomness: Probability of making a random move (0.0 = always use MCTS)
        """
        super().__init__()
        self.player: env.Player = player
        self.simulations: int = simulations
        self.c_param: float = c_param
        self.randomness: float = randomness
    
    def _simulate(self, node: MCTSNode) -> float:
        """Run a random simulation from the given node to a terminal state"""
        # Create a temporary environment copy for simulation
        temp_state = env.State(node.state.board.copy())
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
        
        return best_action
    
    def play(self, e: env.Env) -> tuple[env.Observation, env.Action, env.Reward]:
        """Make a move using MCTS"""
        # Introduce randomness if specified
        if np.random.random() < self.randomness:
            return RandomPlayer(self.player).play(e)
        
        obs = e.observe(self.player)
        
        # Create root node from current state
        root = MCTSNode(
            state=env.State(e.state.board.copy()),
            parent=None,
            action=None,
            player=self.player
        )
        
        # If there are legal actions, run MCTS
        if root.untried_actions or root.children:
            chosen_action = self._mcts_search(root)
            
            if chosen_action is not None:
                reward = e.step(chosen_action, self.player)
                return (obs, chosen_action, reward)
        
        # Fallback to random player if no action found
        return RandomPlayer(self.player).play(e)
    
    def name(self) -> str:
        """Return the name of this player"""
        params = f"sims={self.simulations},c={self.c_param:.2f}"
        if self.randomness > 0:
            params += f",rand={self.randomness:.2f}"
        return f"mcts({params})"

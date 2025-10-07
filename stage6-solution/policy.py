from abc import ABC, abstractmethod
from typing import Self
import numpy as np
import math
from pydantic import BaseModel
from scipy.signal import convolve2d
import torch.multiprocessing as mp


import env
import a0inference

C_PARAM = 1.4142

class Policy(BaseModel, ABC):
    @abstractmethod
    def __call__(self, s: env.State) -> np.ndarray: ...

    @classmethod
    def fmt_config(cls, model_dict: dict) -> str:
        # print like this: {policy_class.__name__}(key=value, key=value, ...)
        config_str = ", ".join([f"{key}={value}" for key, value in model_dict.items()])
        return f"{cls.__name__}({config_str})"


class RandomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> np.ndarray:
        legal_mask = s.legal_mask()
        p = legal_mask / np.sum(legal_mask)
        return p


class HumanPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> np.ndarray:
        env.print_state(s)
        print("0 1 2 3 4 5 6")
        legal_mask = s.legal_mask()
        print("legal mask:", legal_mask)
        chosen_action = np.int8(input("Choose action: "))
        action_probs = np.zeros(env.BOARD_XSIZE, dtype=np.float32)
        action_probs[chosen_action] = 1.0
        return action_probs


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

    def __init__(self, depth: int):
        super().__init__(depth=depth)

    def __call__(self, s: env.State) -> np.ndarray:
        # create a new env and set the state
        e = env.Env()
        e.state = s

        _, chosen_action = minimax(e, self.depth, -math.inf, math.inf)

        action_probs = np.zeros(env.BOARD_XSIZE, dtype=np.float32)
        action_probs[chosen_action] = 1.0
        return action_probs

class AlphaZeroNode:
    """Node in the AlphaZero tree"""

    state: env.State
    player: env.Player
    parent: Self | None
    action: env.Action | None
    visits: int
    q_value: float
    prior: float
    children: dict[env.Action, Self]
    untried_actions: list[env.Action]

    def __init__(
        self,
        state: env.State,
        to_play: env.Player,
        parent: Self | None = None,
        action: env.Action | None = None,  # only none for the root node
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.player = to_play  # Player who will make a move from this state
        self.parent = parent
        self.action = action  # Action that led to this node

        self.visits: int = 0
        self.q_value: float = 0.0
        self.prior: float = 0.0
        self.children = {}

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state"""
        return self.state.is_terminal()

    @property
    def u_value(self) -> float:
        """Returns the exploration score of the node based on UCB1 formula"""
        if self.visits == 0:
            return float("inf")
        return C_PARAM * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self) -> Self:
        """Select the best child using UCB1 formula"""
        best_child = None
        best_weight = -math.inf
        for child in self.children.values():
            # remember that the child is of the opposite player, and thus it's q_value is from the perspective of the opponent player
            weight = child.u_value - child.q_value
            if weight > best_weight:
                best_weight = weight
                best_child = child

        return best_child

    def update(self, result: float) -> None:
        """Update node statistics after a simulation"""
        self.visits += 1
        self.wins += result


class AlphaZeroPolicy(Policy):
    """Monte Carlo Tree Search policy"""

    simulations: int
    checkpoint_path: str
    _inference_request_queue: mp.Queue
    _inference_response_queue: mp.Queue
    _worker_id: int

    def __init__(
        self,
        simulations: int,
        inference_request_queue: mp.Queue,
        inference_response_queue: mp.Queue,
        worker_id: int,
        checkpoint_path: str = "live_inference",
    ) -> None:
        """
        Initialize MCTS Policy

        Args:
            checkpoint_path: Path to the checkpoint
            inference_request_queue: Queue for sending inference requests
            inference_response_queue: Queue for receiving inference responses
            worker_id: ID of the worker
        """
        super().__init__(simulations=simulations, checkpoint_path=checkpoint_path)
        self._inference_request_queue = inference_request_queue
        self._inference_response_queue = inference_response_queue
        self._worker_id = worker_id

    def _evaluate_node(self, node: AlphaZeroNode) -> float:
        """Evaluate the node using the neural network"""
        assert not node.is_evaluated

        request = a0inference.InferenceRequest(
            worker_id=self._worker_id,
            state=node.state,
        )
        self._inference_request_queue.put(request)
        response = self._inference_response_queue.get()
        action_probs = response.action_probs
        value = response.value

        node.is_evaluated = True

        for action in node.get_legal_actions():
            e = env.Env()
            e.state = node.state.copy()
            e.step(action)
            node.children[action] = AlphaZeroNode(
                state=e.state,
                parent=node,
                action=action,
                to_play=env.opponent(node.player),
                prior=action_probs[action],
            )

        return value


    def _alpha_zero_search(self, root: AlphaZeroNode) -> np.ndarray:
        # First, evaluate root if not done
        if not root.is_evaluated:
            self._evaluate_node(root)
        
        for _ in range(self.simulations):
            node = root
            path = [node]
            
            # Selection - traverse down using PUCT
            while node.is_evaluated and not node.is_terminal():
                node = node.best_child()
                path.append(node)
            
            # Evaluation - use neural network instead of rollout
            if not node.is_terminal():
                value = self._evaluate_node(node)
            else:
                # Terminal node - actual game outcome
                value = get_terminal_value(node.state)
            
            # Backpropagation - update Q-values instead of win counts
            for node in reversed(path):
                node.visits += 1
                # Update Q-value as running average
                node.q_value = (node.q_value * (node.visits - 1) + value) / node.visits
                # Flip value for opponent's perspective
                value = -value

    def __call__(self, s: env.State) -> np.ndarray:
        """Make a move using AlphaZero MCTS"""
        # Create root node from current state
        root = AlphaZeroNode(
            state=s.copy(), parent=None, action=None, to_play=s.current_player
        )
        best_action = self._alpha_zero_search(root)
        
        action_probs = np.zeros(env.BOARD_XSIZE, dtype=np.float32)
        action_probs[best_action] = 1.0
        return action_probs

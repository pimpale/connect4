from abc import ABC, abstractmethod
from typing import Self
import numpy as np
import math
import random
from pydantic import BaseModel
import torch.multiprocessing as mp
from scipy.signal import convolve2d


import env
import inference

C_PARAM = 1.4142


class Policy(BaseModel, ABC):
    @abstractmethod
    def __call__(self, env: env.Env) -> env.Action: ...

    def name(self) -> str:
        return self.fmt_config(self.model_dump())

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

    def __init__(self, depth: int):
        super().__init__(depth=depth)

    def __call__(self, s: env.State) -> env.Action:
        # create a new env and set the state
        e = env.Env()
        e.state = s

        _, chosen_action = minimax(e, self.depth, -math.inf, math.inf)

        return chosen_action


class NNPolicy(Policy):
    """Policy that sends inference requests to an inference server"""

    checkpoint_path: str
    _inference_request_queue: mp.Queue
    _inference_response_queue: mp.Queue
    _worker_id: int

    def __init__(
        self,
        inference_request_queue: mp.Queue,
        inference_response_queue: mp.Queue,
        worker_id: int,
        checkpoint_path: str = "live_inference",
    ):
        super().__init__(checkpoint_path=checkpoint_path)
        self._inference_request_queue = inference_request_queue
        self._inference_response_queue = inference_response_queue
        self._worker_id = worker_id

    def __call__(self, s: env.State) -> env.Action:
        # Send inference request
        request = inference.InferenceRequest(
            worker_id=self._worker_id,
            state=s,
        )
        self._inference_request_queue.put(request)

        # Wait for response
        response = self._inference_response_queue.get()
        action_probs = response.action_probs

        # Apply legal mask and sample action
        legal_mask = s.legal_mask()
        raw_p = action_probs * legal_mask
        p = raw_p / np.sum(raw_p)
        chosen_action = env.Action(np.random.choice(len(p), p=p))

        return chosen_action


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""

    state: env.State
    player: env.Player
    parent: Self | None
    action: env.Action | None
    visits: int
    wins: float
    children: dict[env.Action, Self]
    untried_actions: list[env.Action]

    def __init__(
        self,
        state: env.State,
        to_play: env.Player,
        parent: Self | None = None,
        action: env.Action | None = None,  # only none for the root node
    ) -> None:
        self.state = state
        self.player = to_play  # Player who will make a move from this state
        self.parent = parent
        self.action = action  # Action that led to this node

        self.visits: int = 0
        self.wins: float = 0.0  # win score from perspective of to_play
        self.children = {}
        self.untried_actions = list(self.state.legal_actions())

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state"""
        return self.state.is_terminal()

    def is_fully_expanded(self) -> bool:
        """Check if all children have been expanded"""
        return len(self.untried_actions) == 0

    @property
    def q_value(self) -> float:
        """Returns the mean action value of taking the action from the parent node"""
        return self.wins / self.visits

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

    def expand(self) -> Self:
        """Expand the tree by creating a new child node"""
        action: env.Action = self.untried_actions.pop()

        # Create a copy of the state and apply the action
        e = env.Env()
        e.state = self.state.copy()
        e.step(action)

        # Create the child node with the opposite player
        child = MCTSNode(
            state=e.state,
            parent=self,
            action=action,
            to_play=env.opponent(self.player),
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

    def __init__(
        self,
        simulations: int = 1000,
    ) -> None:
        """
        Initialize MCTS Policy

        Args:
            simulations: Number of simulations to run per move
        """
        super().__init__(simulations=simulations)

    def _simulate(self, node: MCTSNode) -> float:
        """Run a random simulation from the given node to a terminal state"""
        # Create a temporary environment copy for simulation
        e = env.Env()
        e.state = node.state.copy()

        # Play random moves until the game ends
        while not e.state.is_terminal():
            # Get legal actions
            legal_actions: list[env.Action] = list(e.state.legal_actions())

            # Choose a random action
            action = random.choice(legal_actions)

            # Apply the action
            e.step(action)

        # Return the result from the node's player's perspective
        if env.is_winner(e.state, node.player):
            return 1.0
        elif env.is_winner(e.state, env.opponent(node.player)):
            return -1.0
        else:
            return 0.0

    def _mcts_search(self, root: MCTSNode) -> env.Action:
        """Run MCTS to find the best action"""
        for _ in range(self.simulations):
            node = root

            # Selection: traverse the tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion: add a new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: run a random playout
            result = self._simulate(node)

            # Backpropagation: update statistics
            while node is not None:
                node.update(result)
                node = node.parent
                result = -result

        # Choose the action with the highest visit count (most robust choice)
        best_action: env.Action | None = None
        best_visits: int = -1

        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        if best_action is None:
            raise ValueError("No action found. Terminal state reached.")

        return best_action

    def __call__(self, s: env.State) -> env.Action:
        """Make a move using MCTS"""
        # Create root node from current state
        root = MCTSNode(
            state=s.copy(), parent=None, action=None, to_play=s.current_player
        )
        return self._mcts_search(root)

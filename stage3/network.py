import env

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BOARD_CONV_FILTERS = 128

ACTOR_LR = 1e-4  # Lower lr stabilises training greatly
GAMMA = 0.80  # Discount factor for reward discounting
ENTROPY_BONUS = 0.15


# (Channel, Width, Height)
def reshape_board(o: env.Observation) -> np.ndarray:
    return np.stack([o.board == 1, o.board == 2])

def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


# Convert State to the format needed for neural network input
def state_batch_to_tensor(
    states: list[env.State], device: torch.device
) -> torch.Tensor:
    """Convert a batch of states to tensor format for neural network input"""
    # States have board attribute which is a numpy array
    # We need to convert each board to proper channels (one for each player)
    batch = []
    for state in states:
        # Create two channels: one for current player's pieces, one for opponent's
        current_channel = (state.board == state.current_player).astype(np.float32)
        opponent_channel = (state.board == env.opponent(state.current_player)).astype(np.float32)
        channels = np.stack([current_channel, opponent_channel])
        batch.append(channels)
    
    # Stack into batch and convert to tensor
    batch_array = np.stack(batch)
    return torch.from_numpy(batch_array).to(device)

class Actor(nn.Module):
    def __init__(self, width: int, height: int):
        super(Actor, self).__init__()

        self.board_width = width
        self.board_height = height

        # ============ PART 1 ============
        # TODO: please initialize the following layers:
        # Conv1: 2 input channels, BOARD_CONV_FILTERS output channels, kernel_size 3, input size (width, height), output size (width-2, height-2),  
        # Conv2: BOARD_CONV_FILTERS input channels, BOARD_CONV_FILTERS output channels, input size (width -2, height -2), output size (width-4, height-4)
        # Fc1: Linear Layer, input size = (width-4)*(height-4)*BOARD_CONV_FILTERS, output size = 512
        # Fc2: Linear Layer, input size = 512, output size = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Channels, Width, Height)
        x = x.to(torch.float32)
       # ============ PART 1 ============
        # TODO: please compute the output by running the x through the following layers. 
        # Conv1 -> Relu -> Conv2 -> Relu -> Flatten -> Fc1 -> Relu -> Fc2 -> Softmax
        # Make sure the output has dimensions (Batch, Width)
        pass



def compute_policy_gradient_loss(
    # Current policy network's probability of choosing an action
    # in (Batch, Action)
    pi_theta_given_st: torch.Tensor,
    # One hot encoding of which action was chosen
    # in (Batch, Action)
    a_t: torch.Tensor,
    # Rewards To Go for the chosen action
    # in (Batch,)
    R_t: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes the policy gradient loss for a vector of examples, and reduces with mean.


    https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations

    The standard policy gradient is given by the expected value over trajectories of:

    :math:`\sum_{t=0}^{T} \nabla_{\theta} (\log \pi_{\theta}(a_t|s_t))R_t`
    
    where:
    * :math:`\pi_{\theta}(a_t|s_t)` is the current policy's probability to perform action :math:`a_t` given :math:`s_t`
    * :math:`R_t` is the rewards-to-go from the state at time t to the end of the episode from which it came.
    """
    # ======== PART 3 ========= 
    # TODO: please implement this function
    # first, find the loss whose gradient is equal to the policy gradient.
    # then, find the entropy of the action
    # in order to provide an incentive for exploration, set the final loss to
    # policy_loss - 0.1*entropy
    # return the mean loss.
    # The output should have shape (1,) 

    pass


def train_policygradient(
    actor: Actor,
    actor_optimizer: torch.optim.Optimizer,
    observation_batch: list[env.Observation],
    action_batch: list[env.Action],
    value_batch: list[env.Value],
) -> list[float]:
    # assert that the batch_lengths are the same
    assert len(observation_batch) == len(action_batch)
    assert len(observation_batch) == len(value_batch)

    # get device
    device = deviceof(actor)

    # convert data to tensors on correct device

    # in (Batch, Width, Height)
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)

    # in (Batch,)
    true_value_batch_tensor = torch.tensor(
        value_batch, dtype=torch.float32, device=device
    )

    # in (Batch, Action)
    # this is a one hot encoding of the chosen action.
    chosen_action_tensor = F.one_hot(
        torch.tensor(action_batch).to(device).long(), num_classes=actor.board_width
    )

    # train actor
    # ======== PART 4 ========
    # TODO: please train the actor 
    # the actor loss should be the loss specified in compute_policy_gradient_loss 

    # while policy gradient loss is just a single scalar, we return a list to match the output of the PPO function (in stage 4)
    return [actor_loss.item()]

def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    """
    Computes the gamma discounted reward-to-go for each state in the trajectory.

    Note: In this particular case, we only get rewards at the end of the game,
    but this function assumes that rewards may be given at each step.
    """

    # ======== PART 2 =========
    # TODO: please implement this function
    # trajectory_rewards is a list of rewards for each step in the trajectory
    # assume that we may get a reward at each step
    # the output should be a list of values with the same length as trajectory_rewards
    # the value at each step should be the GAMMA discounted reward-to-go from that step
    pass
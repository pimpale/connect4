import env

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BOARD_CONV_FILTERS = 64

ACTOR_LR = 1e-4  # Lower lr stabilises training greatly
CRITIC_LR = 1e-4  # Lower lr stabilises training greatly
GAMMA = 0.80  # Discount factor for advantage estimation and reward discounting


# (Channel, Width, Height)
def reshape_board(o: env.Observation) -> np.ndarray:
    return np.stack([o.board == 1, o.board == 2])


# output in (Batch, Channel, Width, Height)
def obs_batch_to_tensor(
    o_batch: list[env.Observation], device: torch.device
) -> torch.Tensor:
    # Convert state batch into correct format
    return torch.from_numpy(np.stack([reshape_board(o) for o in o_batch])).to(device)


# output in (Batch, Channel, Width, Height)
def obs_to_tensor(o: env.Observation, device: torch.device) -> torch.Tensor:
    # we need to add a batch axis and then convert into a tensor
    return torch.from_numpy(np.stack([reshape_board(o)])).to(device)


def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device

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



class Critic(nn.Module):
    def __init__(self, width: int, height: int):
        super(Critic, self).__init__()
        self.board_width = width
        self.board_height = height

        # ============ PART 2 ============
        # TODO: please initialize the following layers:
        # Conv1: 2 input channels, BOARD_CONV_FILTERS output channels, kernel_size 3, input size (width, height), output size (width-2, height-2),  
        # Conv2: BOARD_CONV_FILTERS input channels, BOARD_CONV_FILTERS output channels, input size (width -2, height -2), output size (width-4, height-4)
        # Fc1: Linear Layer, input size = (width-4)*(height-4)*BOARD_CONV_FILTERS, output size = 512
        # Fc2: Linear Layer, input size = 512, output size = 1
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Width, Height)
        x = x.to(torch.float32)
        # ============ PART 2 ============
        # TODO: please compute the output by running the x through the following layers. 
        # Conv1 -> Relu -> Conv2 -> Relu -> Fc1 -> Relu -> Fc2
        # Reshape the output to have dimensions (Batch,)
        pass

def compute_policy_gradient_loss(
    # Current policy network's probability of choosing an action
    # in (Batch, Action)
    pi_theta_given_st: torch.Tensor,
    # One hot encoding of which action was chosen
    # in (Batch, Action)
    a_t: torch.Tensor,
    # Advantage of the chosen action
    # in (Batch,)
    A_pi_theta_st_at: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes the policy gradient loss for a vector of examples, and reduces with mean.


    https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations

    The standard policy gradient is given by the expected value over trajectories of:

    :math:`\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)A^{\pi_{\theta}}(s_t, a_t)`
    
    where:
    * :math:`\pi_{\theta}(a_t|s_t)` is the current policy's probability to perform action :math:`a_t` given :math:`s_t`
    * :math:`A^{\pi_{\theta}}(s_t, a_t)` is the current value network's guess of the advantage of action :math:`a_t` at :math:`s_t`
    """
    # ======== PART 5 ========= 
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
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[env.Observation],
    action_batch: list[env.Action],
    advantage_batch: list[env.Advantage],
    value_batch: list[env.Value],
) -> tuple[list[float], list[float]]:
    # assert that the models are on the same device
    assert next(critic.parameters()).device == next(actor.parameters()).device
    # assert that the batch_lengths are the same
    assert len(observation_batch) == len(action_batch)
    assert len(observation_batch) == len(advantage_batch)
    assert len(observation_batch) == len(value_batch)

    # get device
    device = deviceof(critic)

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

    # in (Batch,)
    advantage_batch_tensor = torch.tensor(advantage_batch).to(device)

    # train critic
    # ======== PART 6 ========
    # TODO: please train the critic.
    # the critic loss should be the MSE loss between the critic prediction and the true value


    # train actor
    # ======== PART 6 ========
    # TODO: please train the actor 
    # the actor loss should be the loss specified in compute_policy_gradient_loss 

    # return the respective losses
    return ([float(actor_loss)], [float(critic_loss)])


def compute_advantage(
    critic: Critic,
    trajectory_observations: list[env.Observation],
    trajectory_rewards: list[env.Reward],
) -> list[env.Advantage]:
    """
    Computes advantage using GAE(GAMMA, 1).

    See here for derivation: https://arxiv.org/abs/1506.02438
    """

    # ======== PART 4 =========
    # TODO: please implement this function (use Equation 18 from https://arxiv.org/abs/1506.02438)
    # trajectory_rewards is a list of rewards for each step in the trajectory
    # trajectory_observations is a list of observations for each step in the trajectory
    # assume that we may get a reward at each step
    # assume tht the length of trajectory_rewards is the same as the length of trajectory_observations
    # the output should be a list of advantages with the same length as trajectory_rewards
    # the value at each step should be the GAE advantage from that step
    pass


def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    """
    Computes the gamma discounted reward-to-go for each state in the trajectory.

    Note: In this particular case, we only get rewards at the end of the game,
    but this function assumes that rewards may be given at each step.
    """

    # ======== PART 3 =========
    # TODO: please implement this function
    # trajectory_rewards is a list of rewards for each step in the trajectory
    # assume that we may get a reward at each step
    # the output should be a list of values with the same length as trajectory_rewards
    # the value at each step should be the GAMMA discounted reward-to-go from that step
    pass

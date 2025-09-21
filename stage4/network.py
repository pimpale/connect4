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

PPO_EPS = 0.2  # PPO clipping parameter
PPO_GRAD_DESCENT_STEPS = (
    10  # Number of gradient descent steps to take on the surrogate loss
)


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

        self.conv1 = nn.Conv2d(2, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            BOARD_CONV_FILTERS, BOARD_CONV_FILTERS, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear((width - 4) * (height - 4) * BOARD_CONV_FILTERS, 512)
        self.fc2 = nn.Linear(512, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Channels, Width, Height)
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # flatten everything except for batch
        x = torch.flatten(x, 1)
        # apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output in (Batch, Width)
        output = F.softmax(x, dim=1)
        return output

class Critic(nn.Module):
    def __init__(self, width: int, height: int):
        super(Critic, self).__init__()
        self.board_width = width
        self.board_height = height

        # ============ PART 1 ============
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
        # ============ PART 1 ============
        # TODO: please compute the output by running the x through the following layers. 
        # Conv1 -> Relu -> Conv2 -> Relu -> Fc1 -> Relu -> Fc2
        # Reshape the output to have dimensions (Batch,)
        pass


def compute_ppo_loss(
    # Old policy network's probability of choosing an action
    # in (Batch, Action)
    pi_thetak_given_st: torch.Tensor,
    # Current policy network's probability of choosing an action
    # in (Batch, Action)
    pi_theta_given_st: torch.Tensor,
    # One hot encoding of which action was chosen
    # in (Batch, Action)
    a_t: torch.Tensor,
    # Advantage of the chosen action
    A_pi_theta_given_st_at: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the PPO surrogate loss function for a vector of examples, and reduces with mean.
    """

    # ======== PART 3 ========
    # TODO: Please implement the PPO surrogate loss function
    # Recall that the PPO surrogate loss function is:
    # L_CLIP(theta, theta_k) = E[ min( pi_theta(a|s) / pi_theta_k(a|s) * A_pi_theta_k(a|s), clip(pi_theta(a|s) / pi_theta_k(a|s), 1 - PPO_EPS, 1 + PPO_EPS) * A_pi_theta_k(a|s) ) ]
    # the shape of ppo_loss_per_example should be (Batch,), since it's computed for every example in the batch
    # See the PPO paper for more details: https://arxiv.org/abs1/1707.06347

    # in (Batch,)
    entropy_per_example = -torch.sum(
        torch.log(pi_theta_given_st) * pi_theta_given_st, 1
    )

    # we reward entropy, since excessive certainty indicate the model is 'overfitting'
    loss_per_example = ppo_loss_per_example - 0.1 * entropy_per_example

    # we take the average loss over all examples
    return loss_per_example.mean()


def train_ppo(
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
    device = next(critic.parameters()).device

    # convert data to tensors on correct device

    # in (Batch, Width, Height)
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)

    # in (Batch,)
    true_value_batch_tensor = torch.tensor(
        value_batch, dtype=torch.float32, device=device
    )

    # in (Batch, Action)
    chosen_action_tensor = F.one_hot(
        torch.tensor(action_batch).to(device).long(), num_classes=actor.board_width
    )

    # in (Batch,)
    advantage_batch_tensor = torch.tensor(advantage_batch).to(device)

    # train critic
    # ======== PART 5 ========
    # TODO: please train the critic.
    # the critic loss should be the MSE loss between the critic prediction and the true value




    # train actor

    # Recall that in the PPO algorithm, we need to set theta to the *optimal* theta with respect to the surrogate loss function,
    # as opposed to the standard policy gradient, where we just update theta.

    # Here's what that means:
    # We have a policy network with parameters theta_k
    # We want to find the optimal theta, theta*, that maximizes the surrogate loss function L_CLIP(theta, theta_k)
    # Note: In practice, we just use gradient descent for PPO_GRAD_DESCENT_STEPS steps to approximate theta*, since we can't analytically solve for theta*
    # We then update: theta_k <- theta*

    # The amount theta* can diverge from theta_k is limited by L_CLIP(theta, theta_k).
    # This is because we want to avoid the new policy diverging too far from the old policy, since that can lead to instability.

    # the old_policy_action_probs are the the predictions made by the pre-train-step network that we want to not diverge too far away from
    # in (Batch, Action)
    old_policy_action_probs = actor.forward(observation_batch_tensor).detach()

    actor_losses = []
    # ======== PART 4 ========
    # Please implement the PPO algorithm
    # Recall that the PPO algorithm is:
    # 1. For each example in the batch, compute the PPO surrogate loss L_CLIP(theta, theta_k)
    # 2. Average the losses over the batch
    # 3. Compute the gradient of the average loss with respect to theta
    # 4. Update theta <- theta + alpha * gradient
    # 5. Repeat steps 1-6 for PPO_GRAD_DESCENT_STEPS steps
    # See the PPO paper for more details: https://arxiv.org/abs1/1707.06347

    # return the respective losses
    return (actor_losses, [critic_loss.item()] * PPO_GRAD_DESCENT_STEPS)


def compute_advantage(
    critic: Critic,
    trajectory_observations: list[env.Observation],
    trajectory_rewards: list[env.Reward],
) -> list[env.Advantage]:
    """
    Computes advantage using GAE.

    See here for derivation: https://arxiv.org/abs/1506.02438
    """


    # ======== PART 2 =========
    # TODO: please implement this function
    # trajectory_rewards is a list of rewards for each step in the trajectory
    # assume that we may get a reward at each step
    # the output should be a list of values with the same length as trajectory_rewards
    # the value at each step should be the GAMMA discounted reward-to-go from that step
    
    pass


def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    """
    Computes the gamma discounted reward-to-go for each state in the trajectory.

    Note: In this particular case, we only get rewards at the end of the game,
    but this function assumes that rewards may be given at each step.
    """

    trajectory_len = len(trajectory_rewards)

    v_batch = np.zeros(trajectory_len)

    v_batch[-1] = trajectory_rewards[-1]

    # Use GAMMA to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + GAMMA * v_batch[t + 1]

    return list(v_batch)

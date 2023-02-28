import env

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BOARD_CONV_FILTERS = 32
HIDDEN_LAYER_SIZE = 32

ACTOR_LR = 1e-4  # Lower lr stabilises training greatly
CRITIC_LR = 1e-4  # Lower lr stabilises training greatly
GAMMA = 0.8

def obs_batch_to_tensor(observation_batch:list[env.Observation]) -> torch.Tensor:
    # Convert state batch into correct format
    return torch.as_tensor(np.stack([o.board for o in observation_batch]))

class Critic(nn.Module):
    def __init__(self, width:int, height:int):
        super(Critic, self).__init__()

        self.board_width = width
        self.board_height = height

        self.conv1 = nn.Conv2d(1, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(BOARD_CONV_FILTERS, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.fc1 = nn.Linear((width-4)*(height-4)*BOARD_CONV_FILTERS, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # cast to float32
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # fully connected layers
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, width:int, height:int):
        super(Actor, self).__init__()

        self.board_width = width
        self.board_height = height


        self.conv1 = nn.Conv2d(1, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(BOARD_CONV_FILTERS, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.fc1 = nn.Linear((width-4)*(height-4)*BOARD_CONV_FILTERS, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, width)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # cast to float32
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations
# The standard policy gradient is given by:
# $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)A^{\pi_{\theta}}(s_t, a_t)$
# where:
# * $\pi_{\theta}(a_t|s_t)$ is the current policy's probability to perform action $a_t$ given $s_t$
# * $A^{\pi_{\theta}}(s_t, a_t)$ is the current value network's guess of the advantage of action $a_t$ at $s_t$
def compute_policy_gradient_loss(
        # Current policy network's probability of choosing an action
        # in (Batch, Action)
        pi_theta_given_st: torch.Tensor,
        # One hot encoding of which action was chosen
        # in (Batch, Action)
        a_t: torch.Tensor,
        # Advantage of the chosen action
        A_pi_theta_given_st_at: torch.Tensor,
    ) -> torch.Tensor:
        # in (Batch,)
        pi_theta_given_st_at = torch.sum(pi_theta_given_st*a_t, 1)
        # in (Batch,)
        actionwise_policy_loss = torch.log(pi_theta_given_st_at)*A_pi_theta_given_st_at

        # we take the average loss over all examples
        return actionwise_policy_loss.mean()


def train(
        actor:Actor,
        critic:Critic,
        actor_optimizer:torch.optim.Optimizer,
        critic_optimizer:torch.optim.Optimizer,
        observation_batch: list[env.Observation],
        action_batch: list[env.Action],
        advantage_batch:list[env.Advantage],
        value_batch:list[env.Value],
    ):
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
        observation_batch_tensor = obs_batch_to_tensor(observation_batch).to(device)

        # in (Batch,) 
        true_value_batch_tensor = torch.tensor(value_batch).to(device)

        # in (Batch, Action)
        chosen_action_tensor = F.one_hot(torch.tensor(action_batch), num_classes=actor.board_width).to(device)

        # in (Batch,) 
        advantage_batch_tensor = torch.tensor(advantage_batch).to(device)

        # train critic
        critic_optimizer.zero_grad()
        pred_value_batch_tensor = critic.forward(observation_batch_tensor)
        loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
        loss.backward()
        critic_optimizer.step()

        # train actor
        actor_optimizer.zero_grad()
        action_probs = actor.forward(observation_batch_tensor)
        loss = compute_policy_gradient_loss(action_probs, chosen_action_tensor, advantage_batch_tensor)
        loss.backward()
        actor_optimizer.step()


# computes advantage
def compute_advantage(
    critic:Critic,
    trajectory_observations: list[env.Observation],
    trajectory_rewards: list[env.Reward],
) -> list[env.Advantage]:
    trajectory_len = len(trajectory_rewards)

    assert len(trajectory_observations) == trajectory_len
    assert len(trajectory_rewards) == trajectory_len

    trajectory_advantages = np.zeros(trajectory_len)

    # calculate the value of the state at the end
    last_obs = obs_batch_to_tensor([trajectory_observations[-1]])
    # move to the device
    last_obs.to(next(critic.parameters()).device)
    last_obs_value = critic.forward(last_obs)[0]

    trajectory_advantages[-1] = last_obs_value  + trajectory_rewards[-1]

    # Use GAMMA to decay the advantage 
    for t in reversed(range(trajectory_len- 1)):
        trajectory_advantages[t] = trajectory_rewards[t] + GAMMA * trajectory_advantages[t + 1]

    return list(trajectory_advantages  )

# computes what the critic network should have predicted
def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    trajectory_len= len(trajectory_rewards)

    v_batch = np.zeros(trajectory_len)

    v_batch[-1] = trajectory_rewards[-1]

    # Use GAMMA to decay the advantage 
    for t in reversed(range(trajectory_len- 1)):
        v_batch[t] = trajectory_rewards[t] + GAMMA * v_batch[t + 1]

    return list(v_batch)


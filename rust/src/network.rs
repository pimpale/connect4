use tch::{
    nn::{self, ModuleT},
    IndexOp,
};

use crate::env;

const BOARD_CONV_FILTERS: i64 = 32;
const GAMMA: f32 = 0.95;
const PPO_EPS: f64 = 0.2;

#[derive(Debug)]
pub struct Critic<const WIDTH: usize, const HEIGHT: usize> {
    device: tch::Device,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl<const WIDTH: usize, const HEIGHT: usize> Critic<WIDTH, HEIGHT> {
    pub fn new(vs: &nn::Path) -> Self {
        let conv1 = nn::conv2d(vs, 3, BOARD_CONV_FILTERS, 3, Default::default());
        let conv2 = nn::conv2d(
            vs,
            BOARD_CONV_FILTERS,
            BOARD_CONV_FILTERS,
            3,
            Default::default(),
        );
        let fc1_in_width = (WIDTH as i64 - 4) * (HEIGHT as i64 - 4) * BOARD_CONV_FILTERS;
        let fc1 = nn::linear(vs, fc1_in_width, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 1, Default::default());
        Self {
            device: vs.device(),
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }

    pub fn device(&self) -> tch::Device {
        self.device
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> nn::ModuleT for Critic<WIDTH, HEIGHT> {
    fn forward_t(&self, xs: &tch::Tensor, _: bool) -> tch::Tensor {
        xs.apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            // reshape to (batch_size, -1)
            .flatten(1, -1)
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            // reshape to (batch_size,)
            .flatten(0, -1)
    }
}

#[derive(Debug)]
pub struct Actor<const WIDTH: usize, const HEIGHT: usize> {
    device: tch::Device,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl<const WIDTH: usize, const HEIGHT: usize> Actor<WIDTH, HEIGHT> {
    pub fn new(vs: &nn::Path) -> Self {
        let conv1 = nn::conv2d(vs, 3, BOARD_CONV_FILTERS, 3, Default::default());
        let conv2 = nn::conv2d(
            vs,
            BOARD_CONV_FILTERS,
            BOARD_CONV_FILTERS,
            3,
            Default::default(),
        );
        let fc1_in_width = (WIDTH as i64 - 4) * (HEIGHT as i64 - 4) * BOARD_CONV_FILTERS;
        let fc1 = nn::linear(vs, fc1_in_width, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Self {
            device: vs.device(),
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }

    pub fn device(&self) -> tch::Device {
        self.device
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> nn::ModuleT for Actor<WIDTH, HEIGHT> {
    fn forward_t(&self, xs: &tch::Tensor, _: bool) -> tch::Tensor {
        xs.apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            // reshape to (batch_size, -1)
            .flatten(1, -1)
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            // reshape to (batch_size, 10)
            .view([-1, 10])
    }
}

pub fn obs_batch_to_tensor<const WIDTH: usize, const HEIGHT: usize>(
    o_batch: &[env::Observation<WIDTH, HEIGHT>],
    device: tch::Device,
) -> tch::Tensor {
    let o_batch_data: Vec<bool> = o_batch.iter().flat_map(|o| o.reshape_board()).collect();
    tch::Tensor::of_slice(&o_batch_data)
        .view([-1, 2, WIDTH as i64, HEIGHT as i64])
        .to_device(device)
}

pub fn obs_to_tensor<const WIDTH: usize, const HEIGHT: usize>(
    o: &env::Observation<WIDTH, HEIGHT>,
    device: tch::Device,
) -> tch::Tensor {
    let o_data: Vec<bool> = o.reshape_board();
    tch::Tensor::of_slice(&o_data)
        .view([1, 2, WIDTH as i64, HEIGHT as i64])
        .to_device(device)
}

pub fn ppo2_loss(
    // Old policy network's probability of choosing an action
    // in (Batch, Action)
    pi_thetak_given_st: &tch::Tensor,
    // Current policy network's probability of choosing an action
    // in (Batch, Action)
    pi_theta_given_st: &tch::Tensor,
    // One hot encoding of which action was chosen
    // in (Batch, Action)
    a_t: &tch::Tensor,
    // Advantage of the chosen action
    a_pi_theta_given_st_at: &tch::Tensor,
) -> tch::Tensor {
    let batch_size = pi_theta_given_st.size()[0];

    let pi_theta_given_st_at =
        (pi_theta_given_st * a_t).sum_dim_intlist([1].as_slice(), false, tch::Kind::Float);

    let pi_thetak_given_st_at =
        (pi_thetak_given_st * a_t).sum_dim_intlist([1].as_slice(), false, tch::Kind::Float);

    // the likelihood ratio (used to penalize divergence from the old policy)
    let likelihood_ratio = &pi_theta_given_st_at / &pi_thetak_given_st_at;

    // in (Batch,)
    let ppo2loss_at_t = tch::Tensor::minimum(
        &(&likelihood_ratio * a_pi_theta_given_st_at),
        &(&likelihood_ratio.clamp(1.0 - PPO_EPS, 1.0 + PPO_EPS) * a_pi_theta_given_st_at),
    );

    // in (Batch,)
    let entropy_at_t = -(pi_theta_given_st.log() * pi_theta_given_st).sum_dim_intlist(
        [1].as_slice(),
        false,
        tch::Kind::Float,
    );

    // in (Batch,)
    let total_loss_at_t = -ppo2loss_at_t - entropy_at_t;

    // we take the average loss over all examples
    total_loss_at_t.mean(tch::Kind::Float)
}

pub fn train_ppo<const WIDTH: usize, const HEIGHT: usize>(
    actor: &Actor<WIDTH, HEIGHT>,
    critic: &Critic<WIDTH, HEIGHT>,
    actor_optimizer: &mut tch::nn::Optimizer,
    critic_optimizer: &mut tch::nn::Optimizer,
    observation_batch: &Vec<env::Observation<WIDTH, HEIGHT>>,
    action_batch: &Vec<env::Action>,
    oldpolicy_batch: &Vec<tch::Tensor>,
    advantage_batch: &Vec<env::Advantage>,
    value_batch: &Vec<env::Value>,
) -> (f32, f32) {
    // assert that the models are on the same device
    assert_eq!(
        actor.device(),
        critic.device(),
        "actor and critic must be on the same device"
    );
    // assert that the batch_lengths are the same
    assert_eq!(
        observation_batch.len(),
        action_batch.len(),
        "observation_batch and action_batch must be the same length"
    );
    assert_eq!(
        observation_batch.len(),
        oldpolicy_batch.len(),
        "observation_batch and oldpolicy_batch must be the same length"
    );
    assert_eq!(
        observation_batch.len(),
        advantage_batch.len(),
        "observation_batch and advantage_batch must be the same length"
    );
    assert_eq!(
        observation_batch.len(),
        value_batch.len(),
        "observation_batch and value_batch must be the same length"
    );

    // get device
    let device = actor.device();

    // convert data to tensors on correct device

    // in (Batch, Width, Height)
    let observation_batch_tensor = obs_batch_to_tensor(observation_batch, device);

    // in (Batch,)
    let true_value_batch_tensor = tch::Tensor::of_slice(value_batch)
        .to_kind(tch::Kind::Float)
        .to_device(device);

    // in (Batch, Action)
    let chosen_action_tensor =
        tch::Tensor::of_slice(&action_batch.iter().map(|x| *x as i64).collect::<Vec<i64>>())
            .one_hot(WIDTH as i64)
            .to_kind(tch::Kind::Float)
            .to_device(device);

    // in (Batch, Action)
    let old_policy_action_probs_batch_tensor = tch::Tensor::vstack(oldpolicy_batch.as_slice())
        .to_device(device);

    // in (Batch,)
    let advantage_batch_tensor = tch::Tensor::of_slice(advantage_batch)
        .to_kind(tch::Kind::Float)
        .to_device(device);

    // train critic
    let critic_loss = {
        // zero gradients
        critic_optimizer.zero_grad();

        // get current value
        let current_value = critic.forward_t(&observation_batch_tensor, true);

        // calculate loss
        let critic_loss = tch::Tensor::mse_loss(
            &current_value,
            &true_value_batch_tensor,
            tch::Reduction::Mean,
        );

        // backpropagate
        critic_loss.backward();

        // update weights
        critic_optimizer.step();

        // return loss
        critic_loss
    };

    // train actor
    let actor_loss = {
        // zero gradients
        actor_optimizer.zero_grad();

        // get current policy
        let current_policy = actor.forward_t(&observation_batch_tensor, true);

        // calculate loss
        let actor_loss = ppo2_loss(
            &current_policy,
            &old_policy_action_probs_batch_tensor,
            &chosen_action_tensor,
            &advantage_batch_tensor,
        );

        // backpropagate
        actor_loss.backward();

        // update weights
        actor_optimizer.step();

        // return loss
        actor_loss
    };

    (f32::from(critic_loss), f32::from(actor_loss))
}

fn compute_advantage<const WIDTH: usize, const HEIGHT: usize>(
    critic: &Critic<WIDTH, HEIGHT>,
    trajectory_observations: &Vec<env::Observation<WIDTH, HEIGHT>>,
    trajectory_rewards: &Vec<env::Reward>,
) -> Vec<env::Advantage> {
    let trajectory_len = trajectory_rewards.len();

    assert_eq!(trajectory_observations.len(), trajectory_len);
    assert_eq!(trajectory_rewards.len(), trajectory_len);

    let mut trajectory_advantages = vec![0.0; trajectory_len];

    // calculate the value of the state at the end
    let last_obs = obs_to_tensor(
        &trajectory_observations[trajectory_len - 1],
        critic.device(),
    );
    let last_obs_value: f32 = critic.forward_t(&last_obs, true).i(0).into();

    trajectory_advantages[trajectory_len - 1] =
        last_obs_value + trajectory_rewards[trajectory_len - 1];

    // Use GAMMA to decay the advantage
    for t in (0..trajectory_len - 1).rev() {
        trajectory_advantages[t] = trajectory_rewards[t] + GAMMA * trajectory_advantages[t + 1];
    }

    trajectory_advantages
}

fn compute_value(trajectory_rewards: &Vec<env::Reward>) -> Vec<env::Value> {
    let trajectory_len = trajectory_rewards.len();

    let mut v_batch = vec![0.0; trajectory_len];

    v_batch[trajectory_len - 1] = trajectory_rewards[trajectory_len - 1];

    // Use GAMMA to decay the advantage
    for t in (0..trajectory_len - 1).rev() {
        v_batch[t] = trajectory_rewards[t] + GAMMA * v_batch[t + 1];
    }

    v_batch
}

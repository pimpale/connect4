use crate::env;
use crate::network;
use crate::utils;
use rand::Rng;
use tch::nn::ModuleT;

pub trait Player<const WIDTH: usize, const HEIGHT: usize> {
    /// play one step of the game
    fn play(
        &self,
        s: &mut env::State<WIDTH, HEIGHT>,
    ) -> (
        // the observation
        env::Observation<WIDTH, HEIGHT>,
        // the probability all actions in the action space
        tch::Tensor,
        // the chosen action
        env::Action,
        // the reward for the action
        env::Reward,
    );

    fn name(&self) -> String;
}

pub struct RandomPlayer {
    player: env::Player,
}

impl<const WIDTH: usize, const HEIGHT: usize> Player<WIDTH, HEIGHT> for RandomPlayer {
    fn play(
        &self,
        s: &mut env::State<WIDTH, HEIGHT>,
    ) -> (
        env::Observation<WIDTH, HEIGHT>,
        tch::Tensor,
        env::Action,
        env::Reward,
    ) {
        let legal_moves = s.legal_moves();
        let action = legal_moves[rand::thread_rng().gen_range(0..=legal_moves.len())];
        let (observation, reward) = s.step(self.player, action);
        let prob = tch::Tensor::zeros(&[WIDTH as i64], (tch::Kind::Float, tch::Device::Cpu));
        (observation, prob, action, reward)
    }

    fn name(&self) -> String {
        "RandomPlayer".to_string()
    }
}

pub struct MinimaxPlayer<const WIDTH: usize, const HEIGHT: usize> {
    player: env::Player,
    randomness: f64,
    depth: i64,
}

impl<const WIDTH: usize, const HEIGHT: usize> MinimaxPlayer<WIDTH, HEIGHT>
where
    [(); WIDTH - 4 + 1]:,
    [(); HEIGHT - 4 + 1]:,
    [(); WIDTH - 1 + 1]:,
    [(); HEIGHT - 1 + 1]:,
{
    pub fn new(player: env::Player, depth: i64, randomness: f64) -> Self {
        Self {
            player,
            randomness,
            depth,
        }
    }

    /// use the minimax algorithm to find the best move, searching up to depth
    fn minimax(
        s: &mut env::State<WIDTH, HEIGHT>,
        depth: i64,
        player: env::Player,
    ) -> (env::Reward, usize) {
        if depth == 0 {
            return (Self::heuristic(s.observe(player)), 0);
        }

        let legal_actions = s.legal_moves();

        if legal_actions.len() == 0 {
            return (0.0, 0);
        }

        let mut best_action = legal_actions[0];
        let mut best_reward = f32::NEG_INFINITY;
        for action in legal_actions {
            s.step(player, action);
            let (reward, _) = Self::minimax(s, depth - 1, env::opponent(player));
            s.undo();
            if reward > best_reward {
                best_reward = reward;
                best_action = action;
            }
        }

        (best_reward, best_action)
    }

    const FILTER_HORIZ: [[f32; 4]; 1] = [[1.0, 1.0, 1.0, 1.0]];
    const FILTER_VERT: [[f32; 1]; 4] = [[1.0], [1.0], [1.0], [1.0]];
    const FILTER_DIAG1: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    const FILTER_DIAG2: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];

    /// check the horizontals, verticals, and diagonals of a boardslice of an observation to calculate a rough heuristic
    fn heuristic_slice(boardslice: &[[bool; WIDTH]; HEIGHT]) -> f32 {
        // cast boardslice to f32 array
        let boardslice = utils::cast2d(boardslice);

        let horiz_out = utils::crosscorrelate2d(&boardslice, &Self::FILTER_HORIZ);
        let vert_out = utils::crosscorrelate2d(&boardslice, &Self::FILTER_VERT);
        let diag1_out = utils::crosscorrelate2d(&boardslice, &Self::FILTER_DIAG1);
        let diag2_out = utils::crosscorrelate2d(&boardslice, &Self::FILTER_DIAG2);

        return utils::sum(&horiz_out)
            + utils::sum(&vert_out)
            + utils::sum(&diag1_out)
            + utils::sum(&diag2_out);
    }

    fn heuristic(obs: env::Observation<WIDTH, HEIGHT>) -> env::Reward {
        let score_diff =
            Self::heuristic_slice(&obs.board[0]) - Self::heuristic_slice(&obs.board[1]);
        f32::exp(score_diff) / (1.0 + f32::exp(score_diff))
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> Player<WIDTH, HEIGHT> for MinimaxPlayer<WIDTH, HEIGHT>
where
    [(); WIDTH - 4 + 1]:,
    [(); HEIGHT - 4 + 1]:,
    [(); WIDTH - 1 + 1]:,
    [(); HEIGHT - 1 + 1]:,
{
    fn play(
        &self,
        s: &mut env::State<WIDTH, HEIGHT>,
    ) -> (
        env::Observation<WIDTH, HEIGHT>,
        tch::Tensor,
        env::Action,
        env::Reward,
    ) {
        let (_, action) = Self::minimax(s, self.depth, self.player);
        let (observation, reward) = s.step(self.player, action);
        let prob = tch::Tensor::zeros(&[WIDTH as i64], (tch::Kind::Float, tch::Device::Cpu));
        (observation, prob, action, reward)
    }

    fn name(&self) -> String {
        "MinimaxPlayer".to_string()
    }
}

struct ActorPlayer<const WIDTH: usize, const HEIGHT: usize> {
    actor: network::Actor<WIDTH, HEIGHT>,
    critic: network::Critic<WIDTH, HEIGHT>,
    player: env::Player,
    epoch: i32,
}

impl<const WIDTH: usize, const HEIGHT: usize> ActorPlayer<WIDTH, HEIGHT> {
    fn new(
        actor: network::Actor<WIDTH, HEIGHT>,
        critic: network::Critic<WIDTH, HEIGHT>,
        epoch: i32,
        player: env::Player,
    ) -> Self {
        Self {
            actor,
            critic,
            player,
            epoch,
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> Player<WIDTH, HEIGHT> for ActorPlayer<WIDTH, HEIGHT>
where
    [(); WIDTH - 4 + 1]:,
    [(); HEIGHT - 4 + 1]:,
    [(); WIDTH - 1 + 1]:,
    [(); HEIGHT - 1 + 1]:,
{
    fn play(
        &self,
        s: &mut env::State<WIDTH, HEIGHT>,
    ) -> (
        env::Observation<WIDTH, HEIGHT>,
        tch::Tensor,
        env::Action,
        env::Reward,
    ) {
        let obs = s.observe(self.player);

        let action_probs = self
            .actor
            .forward_t(&network::obs_to_tensor(&obs, self.actor.device()), false)
            .detach()
            .to_device(tch::Device::Cpu);

        let action_entropy = utils::entropy(&action_probs, tch::Kind::Float);
        if action_entropy < 0.001 {
            panic!("Entropy is too low!");
        }

        let legal_mask = s.legal_mask();

        let action_logprobs = action_probs.log_softmax(0, tch::Kind::Float);

        // apply noise to probs
        let noise = 0.1 * utils::gumbel(&[WIDTH as i64], tch::Kind::Float);
        let adjusted_action_probs = action_logprobs.exp() + noise;

        let chosen_action = i64::from(tch::Tensor::argmax(
            &(adjusted_action_probs * legal_mask),
            0,
            false,
        )) as usize;

        let (observation, reward) = s.step(self.player, chosen_action);

        (observation, action_probs, chosen_action, reward)
    }

    fn name(&self) -> String {
        format!("actor_ckpt_{}", self.epoch)
    }
}

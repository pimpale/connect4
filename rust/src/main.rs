#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
pub mod env;
pub mod network;
pub mod player;
pub mod utils;
pub use tch;

use tch::nn::OptimizerConfig;
use tensorboard_rs::summary_writer::SummaryWriter;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BOARD_XSIZE: usize = 7;
    const BOARD_YSIZE: usize = 6;

    let EPISODES_PER_AGENT = 50;
    let TRAIN_EPOCHS = 1000;
    let MODEL_SAVE_INTERVAL = 100;
    let MAKE_OPPONENT_INTERVAL = 1000;
    let SUMMARY_STATS_INTERVAL = 10;
    let RANDOM_SEED = 42;

    let SUMMARY_DIR = "./summary";
    let MODEL_DIR = "./models";

    std::fs::create_dir_all(SUMMARY_DIR)?;
    std::fs::create_dir_all(MODEL_DIR)?;

    let mut writer = SummaryWriter::new(SUMMARY_DIR);

    let device = tch::Device::cuda_if_available();

    let actor_vs = tch::nn::VarStore::new(device);
    let critic_vs = tch::nn::VarStore::new(device);
    let mut actor: network::Actor<BOARD_XSIZE, BOARD_YSIZE> = network::Actor::new(&actor_vs.root());
    let mut critic: network::Actor<BOARD_XSIZE, BOARD_YSIZE> = network::Actor::new(&critic_vs.root());

    let actor_optimizer = tch::nn::Adam::default().build(&actor_vs, 1e-4)?;
    let critic_optimizer = tch::nn::Adam::default().build(&critic_vs, 1e-4)?;

    let opponent_pool: Vec<Box<dyn player::Player<BOARD_XSIZE, BOARD_YSIZE>>> = vec![
        Box::new(player::RandomPlayer::new(env::Player::Player2)),
    ]



    Ok(())
}

#!/usr/bin/env python3
"""
Multiprocessing-based training script for Connect4 using REINFORCE (Policy Gradient).
Architecture:
- 1 Policy Server: Manages the neural network, performs training and inference
- N Rollout Workers: Collect episodes in parallel using RemoteNNPolicy
"""

import math
import torch.multiprocessing as mp
import numpy as np
import logging
import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
import os
import queue
from dataclasses import dataclass
from typing import List, Tuple
import signal
import sys

import env
import network
import policy
import inference

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training hyperparameters
TRAIN_EPOCHS = 500000
MODEL_SAVE_INTERVAL = 100
SUMMARY_STATS_INTERVAL = 10
RANDOM_SEED = 42

# Batch settings for policy server
BATCH_SIZE_FOR_UPDATE = 256  # Update policy after collecting this many episodes

# Directory settings
SUMMARY_DIR = "./summary"


# Message types for inter-process communication
@dataclass
class RolloutBatch:
    """Batch of rollout data from a worker"""

    states: List[env.State]
    actions: List[env.Action]
    rewards: List[float]
    rewards_vs: dict  # opponent_name -> list of rewards
    worker_id: int


@dataclass
class RolloutRequest:
    """Request to collect rollout data"""
    pass


def play_episode(
    nn_policy: policy.NNPolicy, opponent_policy: policy.Policy, nn_player: env.Player
) -> Tuple[List[env.State], List[env.Action], List[float]]:
    """Play a single episode and return trajectory data"""
    e = env.Env()
    current_player = env.PLAYER1

    s_t: List[env.State] = []
    a_t: List[env.Action] = []
    r_t: List[float] = []

    while not e.game_over():
        if nn_player == current_player:
            s = e.state.copy()
            chosen_action = nn_policy(s)
            reward = e.step(chosen_action)
            s_t.append(s)
            a_t.append(chosen_action)
            r_t.append(reward)
        else:
            opponent_action = opponent_policy(e.state)
            e.step(opponent_action)

        current_player = env.opponent(current_player)

    return s_t, a_t, r_t


def rollout_worker(
    worker_id: int,
    rollouts_per_agent: int,
    rollout_request_queue: mp.Queue,
    rollout_response_queue: mp.Queue,
    inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
):
    """
    Rollout worker process that collects episodes.

    Args:
        worker_id: Unique identifier for this worker
        rollout_request_queue: Queue to recieve requests for rollout data from policy server
        rollout_response_queue: Queue to send rollout data to policy server
        inference_request_queue: Queue to send inference requests
        inference_response_queue: Queue to receive inference responses
    """
    # Set random seed for this worker
    np.random.seed(RANDOM_SEED + worker_id)

    # Create RemoteNNPolicy for this worker
    nn_player = policy.NNPolicy(
        inference_request_queue, inference_response_queue, worker_id
    )

    # Create opponent pool
    opponent_pool = [
        policy.MinimaxPolicy(depth=2, randomness=0.1),
        policy.MinimaxPolicy(depth=2, randomness=0.3),
    ]

    while True:
        # await rollout request
        _ = rollout_request_queue.get()

        # Collect episodes
        s_batch: List[env.State] = []
        a_batch: List[env.Action] = []
        r_batch: List[float] = []
        rewards_vs = {}

        for _ in range(rollouts_per_agent):
            # Pick random opponent
            opponent_player = opponent_pool[np.random.randint(len(opponent_pool))]

            # Randomly assign actor to player 1 or 2
            actor_player_identity = (
                env.PLAYER1 if np.random.randint(2) == 0 else env.PLAYER2
            )

            # Play episode
            s_t, a_t, r_t = play_episode(
                nn_player, opponent_player, actor_player_identity
            )

            # Add to batch
            s_batch += s_t
            a_batch += a_t
            r_batch += r_t

            # Track statistics
            opp_name = opponent_player.fmt_config(opponent_player.model_dump())
            total_reward = np.array(r_t).sum()
            if opp_name in rewards_vs:
                rewards_vs[opp_name].append(total_reward)
            else:
                rewards_vs[opp_name] = [total_reward]

        # Send rollout batch to policy server
        rollout_data = RolloutBatch(
            states=s_batch,
            actions=a_batch,
            rewards=r_batch,
            rewards_vs=rewards_vs,
            worker_id=worker_id,
        )
        rollout_response_queue.put(rollout_data)

        logger.info(f"Worker {worker_id}: Sent batch with {len(s_batch)} transitions")


def train_server(
    rollout_request_queues: List[mp.Queue],
    rollout_response_queue: mp.Queue,
    model_update_request_queue: mp.Queue,
    model_update_response_queue: mp.Queue,
    num_workers: int,
    device: str,
):
    """
    Policy server process that manages the neural network, handles inference, and performs training.

    Args:
        rollout_request_queues: List of queues to send requests for rollout data to workers
        rollout_response_queue: Queue to receive rollout data from workers
        model_update_request_queue: Queue to send model update requests to inference server
        model_update_response_queue: Queue to receive model update responses from inference server
        num_workers: Number of worker processes
        device: Device to use for training (cuda if available, else cpu)
    """
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Create directories
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Initialize actor and optimizer
    device = torch.device(device)
    actor = network.Actor(env.BOARD_XSIZE, env.BOARD_YSIZE).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=network.ACTOR_LR)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=SUMMARY_DIR)

    step = 0
    all_rewards_vs = {}
    batches_received = 0

    logger.info(f"Policy Server: Started with {num_workers} workers on device {device}")

    while step < TRAIN_EPOCHS:
        # Send model update to inference server
        model_update_request_queue.put(
            inference.ModelUpdateRequest(state_dict=actor.state_dict(), step=step)
        )
        logger.info(
            f"Policy Server: Sent model update to inference server at step {step}"
        )
        model_update_response_queue.get()
        logger.info(
            f"Policy Server: Received model update response from inference server at step {step}"
        )
        for rollout_request_queue in rollout_request_queues:
            rollout_request_queue.put(RolloutRequest())
        logger.info(f"Policy Server: Sent rollout request to workers at step {step}")

        # process rollout batches until we have a full batch
        rollout_batches = []
        while len(rollout_batches) < num_workers:
            # get a rollout batch
            try:
                rollout_data = rollout_response_queue.get(timeout=0)
                rollout_batches.append(rollout_data)
                batches_received += 1
            except queue.Empty:
                pass

        # Accumulate all data
        s_batch = []
        a_batch = []
        v_batch = []

        for rollout_data in rollout_batches:
            s_batch += rollout_data.states
            a_batch += rollout_data.actions
            v_batch += network.compute_value(rollout_data.rewards)

            # Merge reward statistics
            for opp_name, rewards in rollout_data.rewards_vs.items():
                if opp_name in all_rewards_vs:
                    all_rewards_vs[opp_name] += rewards
                else:
                    all_rewards_vs[opp_name] = rewards

        # Train the model
        actor_losses = network.train_policygradient(
            actor, actor_optimizer, s_batch, a_batch, v_batch
        )

        # Log metrics
        for actor_loss in actor_losses:
            writer.add_scalar("actor_loss", actor_loss, step)

            # Log average rewards against each opponent
            for opponent_name, rewards in all_rewards_vs.items():
                if len(rewards) > 400:
                    avg_reward = np.array(rewards).mean()
                    writer.add_scalar(
                        f"reward_against_{opponent_name}", avg_reward, step
                    )
                    all_rewards_vs[opponent_name] = []

            # Save model periodically
            if step % MODEL_SAVE_INTERVAL == 0:
                inference_checkpoint_path = (
                    f"{SUMMARY_DIR}/nn_model_ep_{step}_actor.ckpt"
                )
                torch.save(actor.state_dict(), inference_checkpoint_path)
                logger.info(f"Policy Server: Saved model checkpoint at step {step}")

            step += 1

        logger.info(
            f"Policy Server: Step {step}, Loss: {actor_losses[0]:.4f}, "
            f"Transitions: {len(s_batch)}"
        )


def main():
    """Main function to set up and run multiprocessing training"""
    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"

    logger.info(f"Starting multiprocessing training on device: {device_str}")
    logger.info(f"Number of CPU cores: {mp.cpu_count()}")

    # Set number of workers to CPU count
    num_workers = mp.cpu_count()

    # Create queues for communication
    rollout_request_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]
    rollout_response_queue = mp.Queue(
        maxsize=num_workers * 2
    )  # Buffer for rollout data
    model_update_request_queue = mp.Queue(100)  # Model update queue
    model_update_response_queue = mp.Queue(100)  # Model update response queue
    inference_request_queue = mp.Queue(
        maxsize=num_workers * 100
    )  # Inference requests from all workers
    inference_response_queues = [
        mp.Queue(maxsize=num_workers * 100) for _ in range(num_workers)
    ]

    # Start train server process
    train_process = mp.Process(
        target=train_server,
        args=(
            rollout_request_queues,
            rollout_response_queue,
            model_update_request_queue,
            model_update_response_queue,
            num_workers,
            device_str,
        ),
    )
    train_process.start()

    # Start inference server process
    inference_server_process = mp.Process(
        target=inference.inference_server,
        args=(
            inference_request_queue,
            inference_response_queues,
            model_update_request_queue,
            model_update_response_queue,
            device_str,
        ),
    )
    inference_server_process.start()

    # Start worker processes
    worker_processes = []
    for i in range(num_workers):
        worker_process = mp.Process(
            target=rollout_worker,
            args=(
                i,
                math.ceil(BATCH_SIZE_FOR_UPDATE / num_workers),
                rollout_request_queues[i],
                rollout_response_queue,
                inference_request_queue,
                inference_response_queues[i],
            ),
        )
        worker_process.start()
        worker_processes.append(worker_process)

    logger.info(
        f"Started {num_workers} rollout workers, 1 train server, and 1 inference server"
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutting down training...")
        train_process.terminate()
        inference_server_process.terminate()
        for p in worker_processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Wait for train and inference servers to complete
        train_process.join()
        inference_server_process.join()
        # Wait for all workers to complete
        for p in worker_processes:
            p.join()

    except KeyboardInterrupt:
        logger.info("\nInterrupted, cleaning up...")
        train_process.terminate()
        for p in worker_processes:
            p.terminate()


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()

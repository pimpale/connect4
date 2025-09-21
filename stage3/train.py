#!/usr/bin/env python3
"""
Multiprocessing-based training script for Connect4 using REINFORCE (Policy Gradient).
Architecture:
- 1 Policy Server: Manages the neural network, performs training and inference
- N Rollout Workers: Collect episodes in parallel using RemoteNNPolicy
"""

import math
import multiprocessing as mp
import numpy as np
import logging
import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
import os
import time
import queue
from dataclasses import dataclass
from typing import List, Tuple
import signal
import sys
import uuid

import env
import network
import policy

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
INFERENCE_BATCH_SIZE = 64  # Batch size for inference requests
INFERENCE_BATCH_TIMEOUT = 0.01  # Max time to wait for a full batch (seconds)

# Directory settings
SUMMARY_DIR = './summary'
MODEL_DIR = './models'

# Message types for inter-process communication
@dataclass
class RolloutBatch:
    """Batch of rollout data from a worker"""
    states: List[env.State]
    actions: List[env.Action]
    values: List[float]
    rewards_vs: dict  # opponent_name -> list of rewards
    worker_id: int

@dataclass
class InferenceRequest:
    """Request for neural network inference"""
    request_id: str
    state: env.State

@dataclass
class InferenceResponse:
    """Response with inference results"""
    request_id: str
    action_probs: np.ndarray

@dataclass
class ModelUpdate:
    """Updated model parameters from server"""
    state_dict: dict
    step: int

@dataclass
class TerminateSignal:
    """Signal to terminate worker"""
    pass


class RemoteNNPolicy:
    """Policy that sends inference requests to the policy server"""
    
    def __init__(self, inference_request_queue: mp.Queue, inference_response_queue: mp.Queue):
        self.inference_request_queue = inference_request_queue
        self.inference_response_queue = inference_response_queue
        
    def __call__(self, s: env.State) -> env.Action:
        # ======== PART 6 ========
        # TODO: Implement remote inference
        # Steps:
        # 1. Generate unique request ID using uuid.uuid4()
        # 2. Create InferenceRequest with the state
        # 3. Put request in inference_request_queue
        # 4. Wait for response with matching ID from inference_response_queue
        #    (Put back responses that don't match our ID)
        # 5. Apply legal mask to action probabilities
        # 6. Sample action from the distribution
        # Return the chosen action
        pass


def play_episode(
    nn_policy: RemoteNNPolicy, 
    opponent_policy: policy.Policy, 
    nn_player: env.Player
) -> Tuple[List[env.State], List[env.Action], List[float], List[float]]:
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

    # Compute value (rewards-to-go)
    v_t = network.compute_value(r_t)

    return s_t, a_t, r_t, v_t


def rollout_worker(
    worker_id: int,
    rollouts_per_agent: int,
    rollout_queue: mp.Queue,
    inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
    terminate_queue: mp.Queue
):
    """
    Rollout worker process that collects episodes.
    
    Args:
        worker_id: Unique identifier for this worker
        rollout_queue: Queue to send rollout data to policy server
        inference_request_queue: Queue to send inference requests
        inference_response_queue: Queue to receive inference responses
        terminate_queue: Queue to receive termination signals
    """
    # Set random seed for this worker
    np.random.seed(RANDOM_SEED + worker_id)
    
    # Create RemoteNNPolicy for this worker
    nn_player = RemoteNNPolicy(inference_request_queue, inference_response_queue)
    
    # Create opponent pool
    opponent_pool = [
        policy.MinimaxPolicy(depth=3, randomness=0.1),
        policy.MinimaxPolicy(depth=3, randomness=0.3),
        policy.MinimaxPolicy(depth=3, randomness=0.5),
    ]
        
    while True:
        # Check for termination signal (non-blocking)
        try:
            msg = terminate_queue.get_nowait()
            if isinstance(msg, TerminateSignal):
                logger.info(f"Worker {worker_id}: Received termination signal")
                break
        except queue.Empty:
            pass
        
        # ======== PART 7 ========
        # TODO: Collect episodes
        # Initialize batch lists:
        # s_batch: List[env.State] = []
        # a_batch: List[env.Action] = []
        # v_batch: List[float] = []
        # rewards_vs = {}
        
        # For each rollout:
        # 1. Pick random opponent from opponent_pool
        # 2. Randomly assign actor to PLAYER1 or PLAYER2
        # 3. Play episode using play_episode()
        # 4. Add episode data to batch lists
        # 5. Track statistics in rewards_vs dict
        
        # Send RolloutBatch to policy server via rollout_queue
        pass


# processess a batch of inference requests and sends the responses back to the inference_response_queue
def process_inference_batch(
    actor: network.Actor,
    inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
    device: torch.device
):
    """Process a batch of inference requests"""    

    # Process inference requests with batching
    inference_batch = []
    try:
        # Try to build a full batch or wait until timeout
        while len(inference_batch) < INFERENCE_BATCH_SIZE:
            # if we have no requests, we must keep waiting
            timeout = INFERENCE_BATCH_TIMEOUT if len(inference_batch) > 0 else None
            request = inference_request_queue.get(timeout=timeout)
            inference_batch.append(request)
    except queue.Empty:
        pass  # Timeout reached or no more requests
    

    # Convert states to tensor batch
    states = [req.state for req in inference_batch]
    state_tensor = network.state_batch_to_tensor(states, device)
    
    # Run inference
    with torch.inference_mode():
        action_probs_batch = actor.forward(state_tensor).cpu().numpy()
    
    # Create responses
    for i, req in enumerate(inference_batch):
        response = InferenceResponse(
            request_id=req.request_id,
            action_probs=action_probs_batch[i]
        )
        inference_response_queue.put(response)


def policy_server(
    rollout_queue: mp.Queue,
    inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
    num_workers: int,
    device: str
):
    """
    Policy server process that manages the neural network, handles inference, and performs training.
    
    Args:
        rollout_queue: Queue to receive rollout data from workers
        inference_request_queue: Queue to receive inference requests
        inference_response_queue: Queue to send inference responses to workers
        terminate_queues: List of queues to send termination signals to workers
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
    actor.eval()  # Default to eval mode
    actor_optimizer = optim.Adam(actor.parameters(), lr=network.ACTOR_LR)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=SUMMARY_DIR)
    
    step = 0
    all_rewards_vs = {}
    batches_received = 0
    
    logger.info(f"Policy Server: Started with {num_workers} workers on device {device}")

    while step < TRAIN_EPOCHS:

        # process rollout batches until we have a full batch
        rollout_batches = []
        while len(rollout_batches) < num_workers:            
            # do some inference
            process_inference_batch(actor, inference_request_queue, inference_response_queue, device)

            # get a rollout batch
            try:
                rollout_data = rollout_queue.get(timeout=0)
                rollout_batches.append(rollout_data)
                batches_received += 1
            except queue.Empty:
                pass


        # ======== PART 8 ========
        # TODO: Accumulate data from all rollout batches
        # Create lists:
        # accumulated_states = []
        # accumulated_actions = []
        # accumulated_values = []
        
        # For each rollout_data in rollout_batches:
        # 1. Add states, actions, values to accumulated lists
        # 2. Merge rewards_vs statistics into all_rewards_vs dict
        
        # Train the model:
        # 1. Set actor to train mode
        # 2. Call network.train_policygradient()
        # 3. Set actor back to eval mode
        
        # Log metrics:
        # 1. Log actor losses to tensorboard
        # 2. Log average rewards against each opponent (if > 400 samples)
        # 3. Save model checkpoint every MODEL_SAVE_INTERVAL steps
        # 4. Increment step counter
        
        pass


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
    rollout_queue = mp.Queue(maxsize=num_workers * 2)  # Buffer for rollout data
    inference_request_queue = mp.Queue(maxsize=num_workers * 100)  # Inference requests from all workers
    inference_response_queue = mp.Queue(maxsize=num_workers * 100)  # Shared response queue for all workers
    terminate_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]  # Termination signals
    
    # Start policy server process
    server_process = mp.Process(
        target=policy_server,
        args=(
            rollout_queue,
            inference_request_queue,
            inference_response_queue,
            num_workers,
            device_str
        )
    )
    server_process.start()
    
    # Start worker processes
    worker_processes = []
    for i in range(num_workers):
        worker_process = mp.Process(
            target=rollout_worker,
            args=(
                i,
                math.ceil(BATCH_SIZE_FOR_UPDATE / num_workers),
                rollout_queue,
                inference_request_queue,
                inference_response_queue,
                terminate_queues[i]
            )
        )
        worker_process.start()
        worker_processes.append(worker_process)
    
    logger.info(f"Started {num_workers} rollout workers and 1 policy server")
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutting down training...")
        server_process.terminate()
        for p in worker_processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Wait for server process to complete
        server_process.join()
        
        # Wait for all workers to complete
        for p in worker_processes:
            p.join()
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted, cleaning up...")
        server_process.terminate()
        for p in worker_processes:
            p.terminate()


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()

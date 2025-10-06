from dataclasses import dataclass
import os
import time
import numpy as np
import torch
import logging
import torch.multiprocessing as mp
import queue

import a0network
import env

INFERENCE_BATCH_SIZE = 64  # Batch size for inference requests
INFERENCE_BATCH_TIMEOUT = 0.005  # Max time to wait for a full batch (seconds)

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for neural network inference"""

    worker_id: int
    state: env.State


@dataclass
class InferenceResponse:
    """Response with inference results"""

    value: np.ndarray
    action_probs: np.ndarray


@dataclass
class ModelUpdateRequest:
    """Updated model parameters from server"""

    step: int
    state_dict: dict


@dataclass
class ModelUpdateResponse:
    """Response with updated model parameters"""

    step: int


# processess a batch of inference requests and sends the responses back to the inference_response_queue
def process_inference_batch(
    actor: a0network.AlphaZeroNetwork,
    inference_request_queue: mp.Queue,
    inference_response_queues: list[mp.Queue],
    device: torch.device,
):
    """Process a batch of inference requests"""

    # Process inference requests with batching
    inference_batch = []
    try:
        # Try to build a full batch or wait until timeout
        deadline = time.monotonic() + INFERENCE_BATCH_TIMEOUT
        while len(inference_batch) < INFERENCE_BATCH_SIZE:
            request = inference_request_queue.get(timeout=deadline - time.monotonic())
            inference_batch.append(request)
    except queue.Empty:
        # if there are no requests after the timeout, raise
        if len(inference_batch) == 0:
            raise queue.Empty()

    # Convert states to tensor batch
    states = [req.state for req in inference_batch]
    state_tensor = a0network.state_batch_to_tensor(states, device)

    # Run inference
    with torch.inference_mode():
        action_probs_batch_tensor, value_batch_tensor = actor.forward(state_tensor)
        action_probs_batch = action_probs_batch_tensor.cpu().numpy()
        value_batch = value_batch_tensor.cpu().numpy()

    # Create responses
    for i, req in enumerate(inference_batch):
        worker_id = req.worker_id
        response = InferenceResponse(
            action_probs=action_probs_batch[i], value=value_batch[i]
        )
        inference_response_queues[worker_id].put(response)


def inference_server(
    inference_request_queue: mp.Queue,
    inference_response_queues: list[mp.Queue],
    model_update_request_queue: mp.Queue,
    model_update_response_queue: mp.Queue,
    device: torch.device,
):
    """Inference server that processes inference requests and sends responses back.
    The actor is loaded from the checkpoint path and updated when the checkpoint is newer.
    """

    actor = a0network.AlphaZeroNetwork(env.BOARD_XSIZE, env.BOARD_YSIZE)
    actor.to(device)
    actor.eval()

    model_loaded = False

    while True:
        try:
            model_update = model_update_request_queue.get(block=not model_loaded)
        except queue.Empty:
            model_update = None

        if model_update is not None:
            actor.load_state_dict(model_update.state_dict)
            logger.info(
                f"Inference Server: Loaded actor state from checkpoint at step {model_update.step}"
            )
            model_update_response_queue.put(ModelUpdateResponse(step=model_update.step))
            model_loaded = True

        # process inference batch
        try:
            process_inference_batch(
                actor, inference_request_queue, inference_response_queues, device
            )
        except queue.Empty:
            # we use this as an opportunity to wait for a model update
            pass

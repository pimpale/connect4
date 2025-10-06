#!/usr/bin/env python3
"""
Benchmark script for Connect 4 agents.

This script allows testing multiple agents against each other in a matrix format.
Agents are specified in a JSON configuration file.

Usage:
    python benchmark.py -c config.json -n 100  # Run 100 games per matchup
    python benchmark.py -c agents.json -n 500 -w 8  # Use 8 worker processes
"""

import argparse
import json
import torch.multiprocessing as mp
import numpy as np
from typing import Any
import time
import sys
from pathlib import Path
from dataclasses import dataclass
import torch

# Add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

import env
import policy
import inference


@dataclass
class GameResult:
    """Container for a single game result."""

    game_id: int
    player1_name: str
    player2_name: str
    winner: int | None  # env.PLAYER1, env.PLAYER2, or None for draw
    moves: int


def play_single_game(
    game_id: int, policy1: policy.Policy, policy2: policy.Policy
) -> GameResult:
    # Create environment
    e = env.Env()

    # Create policies
    policies = {env.PLAYER1: policy1, env.PLAYER2: policy2}

    moves = 0
    while not e.game_over():
        current_player = e.state.current_player
        active_policy = policies[current_player]

        # Get action from policy
        action = active_policy(e.state)

        # Make the move
        e.step(action)
        moves += 1

    # Determine winner
    winner = e.winner()

    return GameResult(
        game_id=game_id,
        player1_name=policy1.name(),
        player2_name=policy2.name(),
        winner=winner,
        moves=moves,
    )


def rollout_worker(
    worker_games: int,
    result_queue: mp.Queue,
    player1_policy: policy.Policy,
    player2_policy: policy.Policy,
):
    for i in range(worker_games):
        result_queue.put(play_single_game(i, player1_policy, player2_policy))


def play_matchup(
    player1_policy_per_worker: list[policy.Policy],
    player2_policy_per_worker: list[policy.Policy],
    games_per_matchup: int,
    num_workers: int,
) -> list[GameResult]:
    """
    Play a matchup between two players with the specified number of games.
    Half of the games will be played with player1 starting, half with player2 starting.

    Args:
        player1_config: Configuration for player 1
        player2_config: Configuration for player 2
        games_per_matchup: Total number of games to play in this matchup
        num_workers: Number of worker processes

    Returns:
        list of GameResult objects
    """

    result_queue = mp.Queue(maxsize=games_per_matchup)

    # Start num_workers rollout workers
    rollout_workers = [
        mp.Process(
            target=rollout_worker,
            args=(
                games_per_matchup // num_workers
                + (1 if games_per_matchup % num_workers > i else 0),
                result_queue,
                player1_policy_per_worker[i],
                player2_policy_per_worker[i],
            ),
        )
        for i in range(num_workers)
    ]
    for worker in rollout_workers:
        worker.start()

    # turn the result queue into a list
    results = [result_queue.get() for _ in range(games_per_matchup)]
    
    # Wait for all workers to finish
    for worker in rollout_workers:
        worker.join()

    return results


def analyze_matchup_results(results: list[GameResult]) -> dict[str, Any]:
    """
    Analyze results for a specific matchup.

    Args:
        results: list of GameResult objects for this matchup

    Returns:
        dictionary with analysis
    """
    total_games = len(results)
    player1_wins = sum(1 for r in results if r.winner == env.PLAYER1)
    player2_wins = sum(1 for r in results if r.winner == env.PLAYER2)
    draws = sum(1 for r in results if r.winner is None)

    avg_moves = np.mean([r.moves for r in results])

    return {
        "total_games": total_games,
        "player1_wins": player1_wins,
        "player2_wins": player2_wins,
        "draws": draws,
        "player1_win_rate": 100 * player1_wins / total_games if total_games > 0 else 0,
        "player2_win_rate": 100 * player2_wins / total_games if total_games > 0 else 0,
        "draw_rate": 100 * draws / total_games if total_games > 0 else 0,
        "avg_moves": avg_moves,
    }


@dataclass
class InferenceServerData:
    process: mp.Process
    request_queue: mp.Queue
    response_queues: list[mp.Queue]


def create_policy_per_workers(
    configs: list[dict[str, Any]], num_workers: int
) -> tuple[list[list[policy.Policy], list[InferenceServerData]]]:
    """Create policies for each worker, deduplicating checkpoint paths.

    Args:
        configs: list of policy configurations
        num_workers: Number of worker processes

    Returns:
        list of lists where each inner list contains policies for one worker
    """
    # Map to store inference servers by checkpoint path
    inference_servers = {}

    # Create policies for each configuration
    policies_per_config = []

    for config in configs:
        policy_type = config.get("type")
        worker_policies = []

        if policy_type == "RandomPolicy":
            # Create RandomPolicy for each worker
            for _ in range(num_workers):
                worker_policies.append(policy.RandomPolicy())

        elif policy_type == "MinimaxPolicy":
            # Create MinimaxPolicy for each worker
            depth = config.get("depth", 4)
            for _ in range(num_workers):
                worker_policies.append(
                    policy.MinimaxPolicy(depth=depth)
                )

        elif policy_type == "NNCheckpointPolicy":
            checkpoint_path = config.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("NNCheckpointPolicy requires checkpoint_path")

            # Check if we already have a server for this checkpoint
            if checkpoint_path not in inference_servers:
                # Start a new inference server for this checkpoint
                server_data = start_inference_server(checkpoint_path, num_workers)
                inference_servers[checkpoint_path] = server_data
            else:
                server_data = inference_servers[checkpoint_path]

            # Create NNPolicy for each worker using the same server
            for worker_id in range(num_workers):
                worker_policies.append(
                    policy.NNPolicy(
                        checkpoint_path=checkpoint_path,
                        inference_request_queue=server_data.request_queue,
                        inference_response_queue=server_data.response_queues[worker_id],
                        worker_id=worker_id,
                    )
                )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        policies_per_config.append(worker_policies)

    # Store inference servers for cleanup
    return policies_per_config, list(inference_servers.values())


def start_inference_server(
    checkpoint_path: str, num_workers: int
) -> InferenceServerData:
    """Start an inference server for neural network inference.

    Args:
        checkpoint_path: Path to the model checkpoint file
        num_workers: Number of worker processes that will use this server

    Returns:
        InferenceServerData containing the server process and queues
    """
    # Create queues for communication
    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    model_update_request_queue = mp.Queue()
    model_update_response_queue = mp.Queue()

    # Start the inference server process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process = mp.Process(
        target=inference.inference_server,
        args=(
            request_queue,
            response_queues,
            model_update_request_queue,
            model_update_response_queue,
            device,
        ),
    )
    process.start()

    # Load initial model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_update_request_queue.put(
        inference.ModelUpdateRequest(
            step=checkpoint.get("step", 0),
            state_dict=checkpoint,
        )
    )

    # Wait for model to be loaded
    response = model_update_response_queue.get()
    print(f"Inference server started with checkpoint at step {response.step}")

    return InferenceServerData(
        process=process,
        request_queue=request_queue,
        response_queues=response_queues,
    )


def cleanup_inference_servers(inference_servers: list[InferenceServerData]) -> None:
    """Clean up all active inference servers."""
    for server_data in inference_servers:
        server_data.process.terminate()
        server_data.process.join()


def run_tournament(
    agent_configs: list[dict], games_per_matchup: int, num_workers: int
) -> None:
    """
    Run a tournament with all agents playing against each other.

    Args:
        agent_configs: list of agent configurations
        games_per_matchup: Number of games per matchup
        num_workers: Number of worker processes
    """
    num_agents = len(agent_configs)

    print(f"Running tournament with {num_agents} agents")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Using {num_workers} worker processes")
    print("-" * 80)

    try:
        # Create all policies for all agents upfront (deduplicates checkpoint paths)
        print("\nInitializing policies...")
        all_policies, inference_servers = create_policy_per_workers(
            agent_configs, num_workers
        )
        print("Policies initialized successfully.")

        agent_names = [policies[0].name() for policies in all_policies]

        # Display agents
        print("\nAgents in tournament:")
        for i, name, config in zip(range(num_agents), agent_names, agent_configs):
            print(f"  {i + 1}. {name}: {config}")
        print("-" * 80)

        # Run all matchups (each agent plays against every other agent)
        all_results = []
        matchup_results = {}

        start_time = time.time()

        for i, player1_name, player1_policies in zip(
            range(num_agents), agent_names, all_policies
        ):
            for j, player2_name, player2_policies in zip(
                range(num_agents), agent_names, all_policies
            ):
                if i == j:
                    continue  # Don't play against self

                print(
                    f"\nPlaying: {player1_name} vs {player2_name}...",
                    end="",
                    flush=True,
                )

                # Play the matchup
                results = play_matchup(
                    player1_policies, player2_policies, games_per_matchup, num_workers
                )
                all_results.extend(results)

                # Analyze this matchup
                matchup_key = f"{player1_name} vs {player2_name}"
                matchup_results[matchup_key] = analyze_matchup_results(results)

                # Quick summary
                analysis = matchup_results[matchup_key]
                print(
                    f" Done! {player1_name} won {analysis['player1_wins']}/{analysis['total_games']} "
                    + f"({analysis['player1_win_rate']:.1f}%)"
                )

        elapsed_time = time.time() - start_time
    finally:
        # Clean up inference servers
        cleanup_inference_servers(inference_servers)

    # Print detailed results
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)

    total_games = len(all_results)
    print(f"Total games played: {total_games}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Games per second: {total_games / elapsed_time:.2f}")

    # Print matchup matrix
    print("\n" + "-" * 80)
    print("WIN RATE MATRIX (row vs column, %)")
    print("-" * 80)

    # Print header
    max_name_len = max(len(name) for name in agent_names)
    print(" " * (max_name_len + 2), end="")
    for name in agent_names:
        print(f"{name:>15}", end="")
    print()

    # Print matrix rows
    for i, row_name in enumerate(agent_names):
        print(f"{row_name:<{max_name_len}}  ", end="")
        for j, col_name in enumerate(agent_names):
            if i == j:
                print(f"{'---':>15}", end="")
            else:
                matchup_key = f"{row_name} vs {col_name}"
                if matchup_key in matchup_results:
                    win_rate = matchup_results[matchup_key]["player1_win_rate"]
                    print(f"{win_rate:>14.1f}%", end="")
                else:
                    print(f"{'N/A':>15}", end="")
        print()

    # Calculate overall statistics for each agent
    print("\n" + "-" * 80)
    print("OVERALL AGENT PERFORMANCE")
    print("-" * 80)

    agent_stats = {}
    for name in agent_names:
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_games_played = 0

        # Count wins as player 1
        for key, result in matchup_results.items():
            if key.startswith(f"{name} vs"):
                total_wins += result["player1_wins"]
                total_losses += result["player2_wins"]
                total_draws += result["draws"]
                total_games_played += result["total_games"]
            elif f"vs {name}" in key:
                # This agent was player 2
                total_wins += result["player2_wins"]
                total_losses += result["player1_wins"]
                total_draws += result["draws"]
                total_games_played += result["total_games"]

        if total_games_played > 0:
            win_rate = 100 * total_wins / total_games_played
            loss_rate = 100 * total_losses / total_games_played
            draw_rate = 100 * total_draws / total_games_played

            agent_stats[name] = {
                "wins": total_wins,
                "losses": total_losses,
                "draws": total_draws,
                "total": total_games_played,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "draw_rate": draw_rate,
            }

    # Sort by win rate
    sorted_agents = sorted(
        agent_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True
    )

    print(
        f"{'Rank':<6} {'Agent':<{max_name_len}} {'Wins':<10} {'Losses':<10} {'Draws':<10} "
        + f"{'Total':<10} {'Win %':<10} {'Loss %':<10} {'Draw %':<10}"
    )
    print("-" * 100)

    for rank, (name, stats) in enumerate(sorted_agents, 1):
        print(
            f"{rank:<6} {name:<{max_name_len}} {stats['wins']:<10} {stats['losses']:<10} "
            + f"{stats['draws']:<10} {stats['total']:<10} "
            + f"{stats['win_rate']:<10.1f} {stats['loss_rate']:<10.1f} {stats['draw_rate']:<10.1f}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Connect 4 agents in a tournament format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration file format (JSON):
[
    {
        "type": "RandomPolicy"
    },
    {
        "type": "MinimaxPolicy",
        "depth": 4,
        "randomness": 0.0
    },
    {
        "type": "NNCheckpointPolicy",
        "checkpoint_path": "./summary/nn_model_ep_0_actor.ckpt"
    }
]

Examples:
  %(prog)s -c config.json -n 100
      Run tournament with 100 games per matchup using agents from config.json
      
  %(prog)s -c agents.json -n 50 -w 8
      Run tournament with 50 games per matchup using 8 worker processes
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file containing agent definitions",
    )

    parser.add_argument(
        "-n",
        "--num-games",
        type=int,
        required=True,
        help="Number of games per matchup (each pair plays this many games)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            agent_configs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate configuration
    if not isinstance(agent_configs, list):
        print(
            "Error: Configuration must be a list of agent configurations",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(agent_configs) < 2:
        print("Error: At least 2 agents are required for a tournament", file=sys.stderr)
        sys.exit(1)

    for i, config in enumerate(agent_configs):
        if not isinstance(config, dict):
            print(
                f"Error: Agent configuration {i} must be a dictionary", file=sys.stderr
            )
            sys.exit(1)

        if "type" not in config:
            print(
                f"Error: Agent configuration {i} missing 'type' field", file=sys.stderr
            )
            sys.exit(1)

    # Run tournament
    run_tournament(agent_configs, args.num_games, args.workers)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
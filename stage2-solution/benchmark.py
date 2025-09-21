#!/usr/bin/env python3
"""
Benchmark script for Connect 4 minimax player.

Usage:
    python benchmark.py -n 100  # Run 100 simulations
    python benchmark.py -n 1000 -c depth=5 randomness=0.1  # Run 1000 games with depth=5, randomness=0.1
    python benchmark.py -n 500 -c depth=3  # Run 500 games with depth=3
"""

import argparse
import multiprocessing as mp
import numpy as np
from typing import Dict, Any, Tuple
import time
import sys
from pathlib import Path

# Add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

import env
import player


def parse_kwargs(kwargs_str: str) -> Dict[str, Any]:
    """Parse keyword arguments from string format 'key1=value1 key2=value2'."""
    if not kwargs_str:
        return {}
    
    kwargs = {}
    for item in kwargs_str.split():
        if '=' not in item:
            raise ValueError(f"Invalid argument format: {item}. Expected key=value")
        key, value = item.split('=', 1)
        
        # Try to parse the value as different types
        try:
            # Try integer first
            kwargs[key] = int(value)
        except ValueError:
            try:
                # Try float
                kwargs[key] = float(value)
            except ValueError:
                # Keep as string
                kwargs[key] = value
    
    return kwargs


def play_single_game(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Play a single game between minimax and random player.
    
    Args:
        args: Tuple of (game_id, minimax_kwargs)
    
    Returns:
        Dictionary with game results
    """
    game_id, minimax_kwargs = args
    
    # Create environment
    e = env.Env(dims=(6, 7))  # Standard Connect 4 board
    
    # Default minimax parameters
    default_kwargs = {'depth': 4, 'randomness': 0.0}
    default_kwargs.update(minimax_kwargs)
    
    # Randomly decide who goes first
    minimax_first = np.random.choice([True, False])
    
    if minimax_first:
        player1 = player.MinimaxPlayer(env.PLAYER1, **default_kwargs)
        player2 = player.RandomPlayer(env.PLAYER2)
        minimax_player_num = env.PLAYER1
    else:
        player1 = player.RandomPlayer(env.PLAYER1)
        player2 = player.MinimaxPlayer(env.PLAYER2, **default_kwargs)
        minimax_player_num = env.PLAYER2
    
    # Play the game
    current_player = env.PLAYER1
    players = {env.PLAYER1: player1, env.PLAYER2: player2}
    
    moves = 0
    while not e.game_over():
        active_player = players[current_player]
        _ = active_player.play(e)
        moves += 1
        current_player = env.opponent(current_player)
    
    # Determine result
    winner = e.winner()
    
    result = {
        'game_id': game_id,
        'minimax_player': minimax_player_num,
        'minimax_first': minimax_first,
        'winner': winner,
        'moves': moves,
        'minimax_won': winner == minimax_player_num,
        'draw': winner is None
    }
    
    return result


def run_benchmark(num_games: int, minimax_kwargs: Dict[str, Any], num_workers: int = None) -> None:
    """Run benchmark simulations.
    
    Args:
        num_games: Number of games to simulate
        minimax_kwargs: Keyword arguments for MinimaxPlayer
        num_workers: Number of worker processes (default: number of CPU cores)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Running {num_games} games using {num_workers} workers...")
    print(f"Minimax parameters: {minimax_kwargs if minimax_kwargs else 'defaults (depth=4, randomness=0.0)'}")
    print("-" * 60)
    
    # Prepare arguments for each game
    game_args = [(i, minimax_kwargs) for i in range(num_games)]
    
    # Run games in parallel
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(play_single_game, game_args)
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    minimax_wins = sum(1 for r in results if r['minimax_won'])
    random_wins = sum(1 for r in results if not r['draw'] and not r['minimax_won'])
    draws = sum(1 for r in results if r['draw'])
    
    minimax_wins_as_first = sum(1 for r in results if r['minimax_won'] and r['minimax_first'])
    minimax_wins_as_second = sum(1 for r in results if r['minimax_won'] and not r['minimax_first'])
    
    games_as_first = sum(1 for r in results if r['minimax_first'])
    games_as_second = num_games - games_as_first
    
    avg_moves = np.mean([r['moves'] for r in results])
    
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total games played: {num_games}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Games per second: {num_games/elapsed_time:.2f}")
    print()
    print(f"Minimax wins: {minimax_wins} ({100*minimax_wins/num_games:.1f}%)")
    print(f"Random wins:  {random_wins} ({100*random_wins/num_games:.1f}%)")
    print(f"Draws:        {draws} ({100*draws/num_games:.1f}%)")
    print()
    print(f"Minimax performance:")
    print(f"  As Player 1 (first):  {minimax_wins_as_first}/{games_as_first} " +
          f"({100*minimax_wins_as_first/games_as_first:.1f}% win rate)")
    print(f"  As Player 2 (second): {minimax_wins_as_second}/{games_as_second} " +
          f"({100*minimax_wins_as_second/games_as_second:.1f}% win rate)")
    print()
    print(f"Average game length: {avg_moves:.1f} moves")
    print("=" * 60)
    
    # Return results for potential further processing
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Connect 4 minimax player against random player',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -n 100
      Run 100 games with default minimax settings
      
  %(prog)s -n 1000 -c "depth=5 randomness=0.1"
      Run 1000 games with depth=5 and 10%% randomness
      
  %(prog)s -n 500 -c "depth=3" -w 8
      Run 500 games with depth=3 using 8 worker processes
        """
    )
    
    parser.add_argument(
        '-n', '--num-games',
        type=int,
        required=True,
        help='Number of games to simulate'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='',
        help='Minimax player configuration as "key1=value1 key2=value2". ' +
             'Available options: depth (int), randomness (float)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: number of CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Parse minimax configuration
    try:
        minimax_kwargs = parse_kwargs(args.config)
    except ValueError as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Validate configuration
    valid_keys = {'depth', 'randomness'}
    invalid_keys = set(minimax_kwargs.keys()) - valid_keys
    if invalid_keys:
        print(f"Error: Invalid configuration keys: {', '.join(invalid_keys)}", file=sys.stderr)
        print(f"Valid keys are: {', '.join(valid_keys)}", file=sys.stderr)
        sys.exit(1)
    
    # Run benchmark
    try:
        run_benchmark(args.num_games, minimax_kwargs, args.workers)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

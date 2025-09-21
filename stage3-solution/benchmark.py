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
import multiprocessing as mp
import numpy as np
from typing import Dict, Any, Tuple, List
import time
import sys
from pathlib import Path
from dataclasses import dataclass

# Add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

import env
import player


@dataclass
class GameResult:
    """Container for a single game result."""
    game_id: int
    player1_name: str
    player2_name: str
    winner: int | None  # env.PLAYER1, env.PLAYER2, or None for draw
    moves: int


def create_player_from_config(config: Dict[str, Any], player_num: env.Player) -> player.Player:
    """
    Create a player instance from a configuration dictionary.
    
    Args:
        config: Dictionary containing 'type' and optional parameters
        player_num: Which player number (env.PLAYER1 or env.PLAYER2)
    
    Returns:
        Player instance
    
    Raises:
        ValueError: If player type is unknown or parameters are invalid
    """
    player_type = config.get('type')
    
    if player_type == 'RandomPlayer':
        # RandomPlayer only needs the player number
        return player.RandomPlayer(player_num)
    
    elif player_type == 'MinimaxPlayer':
        # MinimaxPlayer needs depth and randomness
        depth = config.get('depth', 4)  # Default depth of 4
        randomness = config.get('randomness', 0.0)  # Default no randomness
        return player.MinimaxPlayer(player_num, depth=depth, randomness=randomness)
    
    elif player_type == 'ActorCheckpointPlayer':
        # ActorCheckpointPlayer needs checkpoint path
        checkpoint_path = config.get('checkpoint_path')  # Default checkpoint path
        board_xsize = config.get('board_xsize', 7)  # Default board xsize
        board_ysize = config.get('board_ysize', 6)  # Default board ysize
        return player.ActorCheckpointPlayer(checkpoint_path, player_num, board_xsize, board_ysize)
    
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def play_single_game(args: Tuple[int, Dict[str, Any], Dict[str, Any], bool]) -> GameResult:
    """
    Play a single game between two agents.
    
    Args:
        args: Tuple of (game_id, player1_config, player2_config, player1_starts)
    
    Returns:
        GameResult object with the outcome
    """
    game_id, player1_config, player2_config, player1_starts = args
    
    # Create environment
    e = env.Env(dims=(6, 7))  # Standard Connect 4 board
    
    # Create players based on who starts
    if player1_starts:
        agent1 = create_player_from_config(player1_config, env.PLAYER1)
        agent2 = create_player_from_config(player2_config, env.PLAYER2)
    else:
        # Swap the agents
        agent1 = create_player_from_config(player2_config, env.PLAYER1)
        agent2 = create_player_from_config(player1_config, env.PLAYER2)
    
    # Play the game
    current_player = env.PLAYER1
    players = {env.PLAYER1: agent1, env.PLAYER2: agent2}
    
    moves = 0
    while not e.game_over():
        active_player = players[current_player]
        _ = active_player.play(e)
        moves += 1
        current_player = env.opponent(current_player)
    
    # Determine winner
    winner = e.winner()
    
    # Adjust winner based on who started
    if winner is not None:
        if player1_starts:
            # No adjustment needed
            pass
        else:
            # Swap winner if player positions were swapped
            winner = env.opponent(winner)
    
    return GameResult(
        game_id=game_id,
        player1_name=agent1.name(),
        player2_name=agent2.name(),
        winner=winner,
        moves=moves
    )


def play_matchup(
    player1_config: Dict[str, Any],
    player2_config: Dict[str, Any],
    games_per_matchup: int,
    num_workers: int = None
) -> List[GameResult]:
    """
    Play a matchup between two players with the specified number of games.
    Half of the games will be played with player1 starting, half with player2 starting.
    
    Args:
        player1_config: Configuration for player 1
        player2_config: Configuration for player 2
        games_per_matchup: Total number of games to play in this matchup
        num_workers: Number of worker processes
    
    Returns:
        List of GameResult objects
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    
    # Prepare arguments for each game
    # Half games with player1 starting, half with player2 starting
    game_args = []
    games_as_first = games_per_matchup // 2
    games_as_second = games_per_matchup - games_as_first
    
    for i in range(games_as_first):
        game_args.append((i, player1_config, player2_config, True))
    
    for i in range(games_as_second):
        game_args.append((games_as_first + i, player1_config, player2_config, False))
    
    # Shuffle to mix up the order
    np.random.shuffle(game_args)
    
    # Run games in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(play_single_game, game_args)
    
    return results


def analyze_matchup_results(results: List[GameResult]) -> Dict[str, Any]:
    """
    Analyze results for a specific matchup.
    
    Args:
        results: List of GameResult objects for this matchup
    
    Returns:
        Dictionary with analysis
    """
    total_games = len(results)
    player1_wins = sum(1 for r in results if r.winner == env.PLAYER1)
    player2_wins = sum(1 for r in results if r.winner == env.PLAYER2)
    draws = sum(1 for r in results if r.winner is None)
    
    avg_moves = np.mean([r.moves for r in results])
    
    return {
        'total_games': total_games,
        'player1_wins': player1_wins,
        'player2_wins': player2_wins,
        'draws': draws,
        'player1_win_rate': 100 * player1_wins / total_games if total_games > 0 else 0,
        'player2_win_rate': 100 * player2_wins / total_games if total_games > 0 else 0,
        'draw_rate': 100 * draws / total_games if total_games > 0 else 0,
        'avg_moves': avg_moves
    }


def run_tournament(
    agent_configs: List[Dict[str, Any]],
    games_per_matchup: int,
    num_workers: int = None
) -> None:
    """
    Run a tournament with all agents playing against each other.
    
    Args:
        agent_configs: List of agent configurations
        games_per_matchup: Number of games per matchup
        num_workers: Number of worker processes
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    num_agents = len(agent_configs)
    
    print(f"Running tournament with {num_agents} agents")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Using {num_workers} worker processes")
    print("-" * 80)
    
    # Display agents
    print("\nAgents in tournament:")
    for i, config in enumerate(agent_configs):
        print(f"  {i+1}. {json.dumps(config)}: {config}")
    print("-" * 80)
    
    # Run all matchups (each agent plays against every other agent)
    all_results = []
    matchup_results = {}
    
    start_time = time.time()
    
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue  # Don't play against self
            
            player1_config = agent_configs[i]
            player2_config = agent_configs[j]
            player1_name = json.dumps(player1_config)
            player2_name = json.dumps(player2_config)
            
            print(f"\nPlaying: {player1_name} vs {player2_name}...", end='', flush=True)
            
            # Play the matchup
            results = play_matchup(player1_config, player2_config, games_per_matchup, num_workers)
            all_results.extend(results)
            
            # Analyze this matchup
            matchup_key = f"{player1_name} vs {player2_name}"
            matchup_results[matchup_key] = analyze_matchup_results(results)
            
            # Quick summary
            analysis = matchup_results[matchup_key]
            print(f" Done! {player1_name} won {analysis['player1_wins']}/{analysis['total_games']} " +
                  f"({analysis['player1_win_rate']:.1f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)
    
    total_games = len(all_results)
    print(f"Total games played: {total_games}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Games per second: {total_games/elapsed_time:.2f}")
    
    # Print matchup matrix
    print("\n" + "-" * 80)
    print("WIN RATE MATRIX (row vs column, %)")
    print("-" * 80)
    
    # Create matrix header
    agent_names = [config['name'] for config in agent_configs]
    
    # Print header
    max_name_len = max(len(name) for name in agent_names)
    print(" " * (max_name_len + 2), end='')
    for name in agent_names:
        print(f"{name:>12}", end='')
    print()
    
    # Print matrix rows
    for i, row_name in enumerate(agent_names):
        print(f"{row_name:<{max_name_len}}  ", end='')
        for j, col_name in enumerate(agent_names):
            if i == j:
                print(f"{'---':>12}", end='')
            else:
                matchup_key = f"{row_name} vs {col_name}"
                if matchup_key in matchup_results:
                    win_rate = matchup_results[matchup_key]['player1_win_rate']
                    print(f"{win_rate:>11.1f}%", end='')
                else:
                    print(f"{'N/A':>12}", end='')
        print()
    
    # Calculate overall statistics for each agent
    print("\n" + "-" * 80)
    print("OVERALL AGENT PERFORMANCE")
    print("-" * 80)
    
    agent_stats = {}
    for config in agent_configs:
        name = json.dumps(config)
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_games_played = 0
        
        # Count wins as player 1
        for key, result in matchup_results.items():
            if key.startswith(f"{name} vs"):
                total_wins += result['player1_wins']
                total_losses += result['player2_wins']
                total_draws += result['draws']
                total_games_played += result['total_games']
            elif f"vs {name}" in key:
                # This agent was player 2
                total_wins += result['player2_wins']
                total_losses += result['player1_wins']
                total_draws += result['draws']
                total_games_played += result['total_games']
        
        if total_games_played > 0:
            win_rate = 100 * total_wins / total_games_played
            loss_rate = 100 * total_losses / total_games_played
            draw_rate = 100 * total_draws / total_games_played
            
            agent_stats[name] = {
                'wins': total_wins,
                'losses': total_losses,
                'draws': total_draws,
                'total': total_games_played,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'draw_rate': draw_rate
            }
    
    # Sort by win rate
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    print(f"{'Rank':<6} {'Agent':<{max_name_len}} {'Wins':<10} {'Losses':<10} {'Draws':<10} " +
          f"{'Total':<10} {'Win %':<10} {'Loss %':<10} {'Draw %':<10}")
    print("-" * 80)
    
    for rank, (name, stats) in enumerate(sorted_agents, 1):
        print(f"{rank:<6} {name:<{max_name_len}} {stats['wins']:<10} {stats['losses']:<10} " +
              f"{stats['draws']:<10} {stats['total']:<10} " +
              f"{stats['win_rate']:<10.1f} {stats['loss_rate']:<10.1f} {stats['draw_rate']:<10.1f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Connect 4 agents in a tournament format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration file format (JSON):
[
    {
        "type": "RandomPlayer"
    },
    {
        "type": "MinimaxPlayer",
        "depth": 4,
        "randomness": 0.0
    },
    {
        "type": "MCTSPlayer",
        "simulations": 1000,
        "c_param": 1.4142,
        "randomness": 0.0
    }
]

Examples:
  %(prog)s -c config.json -n 100
      Run tournament with 100 games per matchup using agents from config.json
      
  %(prog)s -c agents.json -n 50 -w 8
      Run tournament with 50 games per matchup using 8 worker processes
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to JSON configuration file containing agent definitions'
    )
    
    parser.add_argument(
        '-n', '--num-games',
        type=int,
        required=True,
        help='Number of games per matchup (each pair plays this many games)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: number of CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Load configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            agent_configs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate configuration
    if not isinstance(agent_configs, list):
        print("Error: Configuration must be a list of agent configurations", file=sys.stderr)
        sys.exit(1)
    
    if len(agent_configs) < 2:
        print("Error: At least 2 agents are required for a tournament", file=sys.stderr)
        sys.exit(1)
    
    for i, config in enumerate(agent_configs):
        if not isinstance(config, dict):
            print(f"Error: Agent configuration {i} must be a dictionary", file=sys.stderr)
            sys.exit(1)
        
        if 'type' not in config:
            print(f"Error: Agent configuration {i} missing 'type' field", file=sys.stderr)
            sys.exit(1)
    
    # Run tournament
    try:
        run_tournament(agent_configs, args.num_games, args.workers)
    except KeyboardInterrupt:
        print("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during tournament: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

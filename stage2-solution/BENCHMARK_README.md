# Connect 4 Benchmark Scripts

This directory contains two benchmark scripts for evaluating the minimax player against a random player.

## Scripts

### 1. benchmark.py
The original benchmark script that uses the existing `env.py` and `player.py` modules. This requires scipy to be installed.


## Usage

Both scripts have the same command-line interface:

```bash
# Run 100 games with default settings (depth=4, randomness=0.0)
python benchmark_standalone.py -n 100

# Run 1000 games with custom minimax parameters
python benchmark_standalone.py -n 1000 -c "depth=5 randomness=0.1"

# Run 500 games with depth=3 using 8 worker processes
python benchmark_standalone.py -n 500 -c "depth=3" -w 8
```

## Command-line Options

- `-n, --num-games`: Number of games to simulate (required)
- `-c, --config`: Minimax player configuration as "key1=value1 key2=value2"
  - `depth`: Search depth for minimax algorithm (default: 4)
  - `randomness`: Probability of making a random move instead of minimax (default: 0.0)
- `-w, --workers`: Number of worker processes for parallel execution (default: number of CPU cores)

## Features

- **Multiprocessing**: Uses all available CPU cores by default for parallel game simulation
- **Detailed Statistics**: Reports win rates, draw rates, and performance as first/second player
- **Configurable Minimax**: Adjust search depth and randomness via command-line arguments
- **Performance Metrics**: Shows games per second and average game length

## Example Output

```
============================================================
BENCHMARK RESULTS
============================================================
Total games played: 100
Time elapsed: 19.62 seconds
Games per second: 5.10

Minimax wins: 100 (100.0%)
Random wins:  0 (0.0%)
Draws:        0 (0.0%)

Minimax performance:
  As Player 1 (first):  51/51 (100.0% win rate)
  As Player 2 (second): 49/49 (100.0% win rate)

Average game length: 11.7 moves
============================================================
```

## Implementation Notes

The minimax player uses:
- Alpha-beta pruning for efficiency
- A heuristic evaluation function when depth limit is reached
- Configurable search depth and randomness parameters

The benchmark randomly assigns which player goes first in each game to ensure fair evaluation.

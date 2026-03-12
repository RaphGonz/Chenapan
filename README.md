# Alpha-Pan

An AlphaZero implementation for Chenapan, a custom 5x5 strategy board game.

## Overview

Alpha-Pan trains a neural network agent to play Chenapan using self-play reinforcement learning, following the AlphaZero algorithm:

1. **Self-play** — the current model plays against itself to generate training data
2. **Training** — a residual neural network learns policy (move probabilities) and value (position evaluation) from self-play games
3. **Iteration** — the improved model replaces the old one and the cycle repeats

## Architecture

- **`Chenapan`** — game logic: board representation, legal moves, win/draw detection, perspective flip
- **`AlphaPanNet`** — residual neural network with a shared tower, policy head (25×25 move probabilities), and value head (scalar position evaluation)
- **`MCTS`** — Monte Carlo Tree Search guided by the neural network prior
- **`AlphaPan`** — training loop: self-play data generation, replay buffer, network optimization

## Training

```bash
python alpha_pan.py
```

Trains for 100 iterations, overwriting `model.pt` and `optim.pt` after each one. Per-iteration logs are printed to console:

```
Iter 000 | PolicyLoss=0.7839 | ValueLoss=0.9753 | WinRate=0.00% | NonWinRate=100.00%
```

To use the trained model in another script without triggering a training run:

```python
from alpha_pan import Chenapan, AlphaPanNet, MCTS, AlphaPan
```

## Playing

```bash
python gui.py
```

Opens a pygame window to play Chenapan against the AI. `model.pt` is loaded automatically on startup (random weights if no checkpoint exists).

- **Click a piece** to select it — valid destinations are highlighted
- **Click a destination** to move
- **Click elsewhere** to deselect
- You play as red (first to move); the AI plays as black

The side panel shows the move counter, draw loop tracker, and a win probability bar updated after each AI move.

When the game ends an overlay shows the result. **Press any key** to restart.

## Piece notation

| Label | Piece |
|-------|-------|
| 0 | Joker (white) |
| A | 1 |
| 2–9 | Face value |
| V | 10 |
| D | 11 |
| R | 12 |

Red disks = player 1 (positive values), black disks = player 2 (negative values).

## Requirements

- Python 3.x
- PyTorch
- NumPy
- tqdm
- pygame-ce (`pip install pygame-ce`)

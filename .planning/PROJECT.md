# Alpha-Pan

## What This Is

Alpha-Pan is an AlphaZero-style self-play reinforcement learning system for Chénapan, a custom 5×5 board game. The AI learns to play by competing against itself, guided by a neural network (policy + value heads) and Monte Carlo Tree Search. The goal is to train a model that plays well, then face it through a simple pygame GUI.

## Core Value

The AI must learn to win, not draw — draws are treated as defeats to force aggressive, decisive play.

## Requirements

### Validated

- ✓ Game engine (5×5 board, all 6 piece types, swap mechanic, win/draw conditions) — existing
- ✓ MCTS with neural network guidance (AlphaZero style, no random rollout) — existing
- ✓ AlphaPanNet: 5-channel convolutional input, policy head (25×25) + value head (scalar) — existing
- ✓ Self-play data collection loop (`selfPlay()`) — existing
- ✓ Training loop with batch SGD (`train()` + `learn()`) — existing
- ✓ Model checkpointing (`model_N.pt`, `optim_N.pt`) — existing
- ✓ Console play mode (human vs AI via terminal input) — existing (commented out)

### Active

- [ ] Draw penalty: draws return -1 (same as losing) instead of 0 — makes the AI strongly prefer winning over drawing
- [ ] Training pipeline: proper training configuration with progress logging and monitoring
- [ ] Pygame GUI: simple window to display the board, click to make moves, play against trained AI

### Out of Scope

- Online multiplayer — single local human vs AI only
- AI vs AI visualization — training is headless, GUI is for play only
- Move suggestion / hints — just play, no assistance

## Context

The entire project lives in `alpha_pan.py`. Game logic, neural network, MCTS, and training are all in one file. The commented-out code at the bottom is a functional console play loop that can serve as the basis for the GUI.

**Draw conditions in the current code:**
- More than 50 moves (`MAX_NUMBER_OF_MOVES = 50`)
- Same board state visited 3+ times (`MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED = 3`)
- No valid moves available

**Known minor issue:** `temperature_action_probs` is computed in `selfPlay()` but the raw `action_probs` is used for sampling — temperature is not applied.

## Constraints

- **Tech stack**: Python, NumPy, PyTorch — no framework changes
- **GUI**: pygame (simple, local window — not web, not tkinter)
- **Hardware**: Runs on CPU if no CUDA available (training will be slow but functional)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Draw value = -1 (not 0) | User wants AI to play aggressively, avoid loops | — Pending |
| Pygame for GUI | Simple window, easy to integrate with existing NumPy board | — Pending |
| Keep everything in one file | Existing structure, low complexity project | — Pending |

---
*Last updated: 2026-03-12 after initialization*

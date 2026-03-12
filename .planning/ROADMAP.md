# Roadmap: Alpha-Pan

## Overview

Alpha-Pan's existing codebase has the full RL pipeline in place but four correctness bugs corrupt the training signal, and there are no hyperparameter values that produce real learning. Phase 1 fixes all four bugs, upgrades the architecture, and configures the training pipeline to produce a genuine checkpoint. Phase 2 builds a pygame GUI on top of that working system so a human can play against the trained AI.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Training Foundation** - Fix all four training bugs, upgrade the network architecture, and configure the pipeline for real learning (completed 2026-03-12)
- [ ] **Phase 2: Pygame GUI** - Build a playable human-vs-AI window with correct rendering, click-to-move, and background AI computation

## Phase Details

### Phase 1: Training Foundation
**Goal**: A correct, observable training pipeline that produces a checkpoint representing genuine learning
**Depends on**: Nothing (first phase)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, PIPE-01, PIPE-02, PIPE-03, PIPE-04, MODEL-01
**Success Criteria** (what must be TRUE):
  1. Training runs to completion with per-iteration policy loss, value loss, and win/draw rate printed to console
  2. Draw outcomes are penalized identically to losses — the value head cannot collapse to predicting zero
  3. Temperature is applied to action sampling during self-play — exploration is live, not dead code
  4. Running the script twice produces identical draw-detection results — board hashing is deterministic
  5. `from alpha_pan import Chenapan, AlphaPanNet, MCTS, AlphaPan` in a separate file does not trigger a training run
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Fix four training bugs: draw value (-1), temperature sampling, deterministic hashing, batch slice off-by-one (completed 2026-03-12)
- [x] 01-02-PLAN.md — Upgrade AlphaPanNet to residual architecture; fix tqdm import, add __main__ guard, add per-iteration logging, set real hyperparameters (completed 2026-03-12)

### Phase 2: Pygame GUI
**Goal**: A playable human-vs-AI window where a human plays Chenapan against the strongest available checkpoint
**Depends on**: Phase 1
**Requirements**: GUI-01, GUI-02, GUI-03, GUI-04, GUI-05
**Success Criteria** (what must be TRUE):
  1. The 5x5 board with all piece values visible renders in a pygame window when `python gui.py` is run
  2. Clicking a piece highlights valid destinations; clicking a destination completes the move
  3. The window remains responsive and redraws during AI computation — no "(Not Responding)" freeze
  4. When the game ends a result screen appears; pressing a key restarts without relaunching the program
  5. The highest-numbered checkpoint is loaded automatically on startup and the model is in eval mode
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Board rendering, piece display (custom notation), click-to-move with valid-move highlighting, side panel with move counter and draw loop tracker (completed 2026-03-12)
- [ ] 02-02-PLAN.md — AI background thread, MCTS.search() value return, checkpoint auto-load with model.eval(), game-over overlay, win probability bar

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Training Foundation | 2/2 | Complete   | 2026-03-12 |
| 2. Pygame GUI | 1/2 | In Progress | - |

---
phase: 01-training-foundation
plan: 02
subsystem: training
tags: [pytorch, alphazero, residual-network, tqdm, self-play]

# Dependency graph
requires:
  - phase: 01-training-foundation/01-01
    provides: draw value fix, board hashing fix, temperature sampling fix, batch slice fix
provides:
  - ResidualBlock class with skip connections enabling gradient flow in deep network
  - AlphaPanNet residual tower (4 blocks, 64 hidden channels) replacing old conv/deconv architecture
  - __main__ guard enabling safe module import without triggering training
  - Per-iteration console logging with PolicyLoss, ValueLoss, WinRate, NonWinRate
  - train() return value (policy_loss_sum, value_loss_sum, num_batches) for aggregation
  - Real training hyperparameters: num_iterations=100, num_selfPlay_iterations=100, num_epochs=4
affects: [gui.py, inference, model checkpoints]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ResidualBlock with skip connection (out += residual) for gradient flow
    - AlphaZero-style network: start_block -> ModuleList of ResidualBlocks -> value_head + policy_head
    - tqdm progress bars wrapping self-play and training epoch loops
    - train() accumulates and returns per-batch loss sums for upstream aggregation

key-files:
  created: []
  modified:
    - alpha_pan.py

key-decisions:
  - "Residual architecture over deconv tower: skip connections prevent vanishing gradients at depth, start_block projects 5 channels to 64 hidden before residual processing"
  - "Policy head outputs flat 25*25 logits then .view(-1, 25, 25) — preserves MCTS contract without changing call sites"
  - "NonWinRate label used instead of DrawRate: outcome -1 is indistinguishable between draw and loss from memory tuple alone"
  - "trange replaces manual print-per-epoch loops: provides progress visibility without noise at scale"

patterns-established:
  - "Safe module import: all execution code gated behind if __name__ == '__main__': guard"
  - "Loss aggregation: train() returns (policy_loss_sum, value_loss_sum, num_batches) tuple, learn() divides by max(num_batches_total, 1) for safe average"
  - "Per-iteration structured logging: Iter 000 | PolicyLoss=X.XXXX | ValueLoss=X.XXXX | WinRate=XX.XX% | NonWinRate=XX.XX%"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, MODEL-01]

# Metrics
duration: 22min
completed: 2026-03-12
---

# Phase 1 Plan 02: Pipeline Hardening and Residual Architecture Summary

**ResidualBlock + AlphaPanNet residual tower with 4 skip-connection blocks replacing deconv architecture, plus __main__ guard, tqdm progress bars, per-iteration structured logging, and production hyperparameters (100 iterations x 100 games)**

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-12T16:22:30Z
- **Completed:** 2026-03-12T16:44:51Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- AlphaPanNet replaced with a 4-block residual tower — gradients flow through skip connections, preventing vanishing gradient collapse at depth
- `__main__` guard and tqdm plain import eliminate ImportError and prevent training on module import, enabling safe use from gui.py
- Per-iteration structured console logging (PolicyLoss, ValueLoss, WinRate, NonWinRate) gives visibility into training progress at scale
- Real production hyperparameters set: num_iterations=100, num_selfPlay_iterations=100, num_epochs=4

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix tqdm import and add __main__ guard** - `bf36917` (feat)
2. **Task 2: Console logging and real hyperparameters** - `6ce9211` (feat)
3. **Task 3: Replace AlphaPanNet with residual tower** - `66396ba` (feat)

## Files Created/Modified
- `alpha_pan.py` — Added ResidualBlock, replaced AlphaPanNet with residual tower, added __main__ guard, updated train() return signature, updated learn() with tqdm loops and per-iteration logging, updated __main__ args to production values

## Decisions Made
- Residual architecture over the old deconv tower: skip connections prevent vanishing gradients at depth. The old compression-deconv approach had no gradient bypass paths through 3 conv layers, making learning at depth unreliable.
- Policy head uses flat Linear(32*5*5, 25*25) then `.view(-1, 25, 25)` instead of ConvTranspose layers — simpler, correct output shape, fully compatible with MCTS.search() masking.
- NonWinRate label used instead of DrawRate: the memory tuple outcome field (-1) cannot distinguish draw from loss without additional game state tracking.
- trange wraps both self-play and training loops for progress visibility without adding per-game print spam.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- System Python (3.14) does not have numpy/torch installed. Anaconda Python (/c/Users/raphg/anaconda3/python.exe) was used for all verifications. This is a pre-existing environment configuration issue, not introduced by this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `python alpha_pan.py` is ready to run a 100-iteration training session with GPU if available
- All Phase 1 success criteria satisfied: safe import, per-iteration logging, residual architecture, real hyperparameters, no tqdm.notebook
- Phase 2 (gui.py) can safely import from alpha_pan.py without triggering training
- Checkpoint files (model_N.pt, optim_N.pt) will be saved each iteration

---
*Phase: 01-training-foundation*
*Completed: 2026-03-12*

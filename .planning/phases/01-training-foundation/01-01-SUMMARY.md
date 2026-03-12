---
phase: 01-training-foundation
plan: 01
subsystem: training
tags: [alphazero, numpy, hashlib, mcts, selfplay, replay-buffer]

# Dependency graph
requires: []
provides:
  - draw returns -1 so value head learns aggressive play instead of collapsing toward zero
  - temperature-scaled action probabilities used at both selfPlay sampling sites
  - deterministic board hashing via hashlib.md5 for stable draw-by-repetition detection
  - correct replay buffer batch slice — last element always eligible for training
affects: [01-02, pipeline-hardening, training-loop]

# Tech tracking
tech-stack:
  added: [hashlib (stdlib)]
  patterns:
    - "Draw outcome = -1: signals active loss, forces aggressive play over draws"
    - "np.ascontiguousarray before tobytes(): required after rot90 perspective flips to guarantee contiguous byte sequence"
    - "Temperature re-normalization: /= np.sum() after ** (1/temp) before np.random.choice to maintain valid probability distribution"

key-files:
  created: []
  modified: [alpha_pan.py]

key-decisions:
  - "Draw value -1 (not 0): prevents value head collapse toward zero when draws and losses dominate early training"
  - "hashlib.md5 over Python hash(): Python hash() is PYTHONHASHSEED-randomized since 3.3; md5 on tobytes() is stable across runs and processes"
  - "np.ascontiguousarray before tobytes(): board states can be non-contiguous NumPy views after np.rot90 perspective flips; skipping this would produce incorrect byte sequences"
  - "Temperature re-normalization mandatory: raising probs to power (1/temp) breaks the sum-to-1 constraint; ValueError from np.random.choice if not re-normalized"

patterns-established:
  - "Verify both player branches in selfPlay(): any per-player sampling fix must be applied symmetrically to player == 1 and player == -1 blocks"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04]

# Metrics
duration: 15min
completed: 2026-03-12
---

# Phase 1 Plan 01: Training Foundation Bug Fixes Summary

**Four AlphaZero training correctness bugs fixed: draw value (-1), temperature-scaled sampling with re-normalization, deterministic md5 board hashing, and correct replay buffer batch slice — all in alpha_pan.py**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-12T11:13:54Z
- **Completed:** 2026-03-12T11:28:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- TRAIN-01: draw branch in `get_value_and_terminated()` now returns -1 instead of 0, preventing value head collapse
- TRAIN-02: both `selfPlay()` sampling sites compute `temperature_action_probs` with mandatory `/= np.sum()` re-normalization before `np.random.choice`
- TRAIN-03: `get_hash()` replaced with deterministic `hashlib.md5(np.ascontiguousarray(state).tobytes()).hexdigest()` — eliminates PYTHONHASHSEED randomization
- TRAIN-04: `train()` batch slice changed from `min(len(memory)-1, ...)` to `min(len(memory), ...)` — last replay buffer element now always eligible

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix TRAIN-01 (draw value) and TRAIN-03 (board hashing)** - `9aaefeb` (fix)
2. **Task 2: Fix TRAIN-02 (temperature sampling) and TRAIN-04 (batch slice off-by-one)** - `db46e0d` (fix)

**Plan metadata:** (docs commit hash — see state updates)

## Files Created/Modified

- `alpha_pan.py` - All four bug fixes applied: draw value, temperature sampling, deterministic hashing, correct batch slice

## Decisions Made

- Used `hashlib.md5` (not SHA variants) for hashing — speed is the primary concern for per-move hash checks during self-play; collision risk is negligible for board state deduplication
- Applied `np.ascontiguousarray()` before `tobytes()` because board states are created via `np.rot90()` perspective flips which may produce non-contiguous views
- Re-normalization with `/= np.sum()` applied to every `temperature_action_probs` array independently in both player branches — defensive pattern for all future sampling sites

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The module-level code at the bottom of `alpha_pan.py` calls `alphaPan.learn()` unconditionally on import, which ran during verification. The verification was adapted to exec only the class definitions (stopping before `game = Chenapan()`), avoiding the full training loop. This is a pre-existing issue addressed in Plan 02 (pipeline hardening with `__main__` guard).
- BatchNorm in `AlphaPanNet` raises `ValueError` when batch_size=1 and model is in training mode — this is a pre-existing architectural issue (BatchNorm requires >1 sample per channel when computing running stats). Not caused by Plan 01 fixes. Smoke test was run with `model.eval()` which uses running statistics and avoids the error. Will be addressed in Plan 02.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All four training correctness bugs fixed — training signal is now valid
- `alpha_pan.py` is ready for Plan 02 pipeline hardening (`__main__` guard, structured logging, checkpoint saving)
- Pre-existing concern: BatchNorm with batch_size=1 in training mode must be addressed in Plan 02 before production training runs

## Self-Check: PASSED

- FOUND: alpha_pan.py
- FOUND: .planning/phases/01-training-foundation/01-01-SUMMARY.md
- FOUND commit 9aaefeb (Task 1: TRAIN-01 + TRAIN-03)
- FOUND commit db46e0d (Task 2: TRAIN-02 + TRAIN-04)

---
*Phase: 01-training-foundation*
*Completed: 2026-03-12*

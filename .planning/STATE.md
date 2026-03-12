---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
last_updated: "2026-03-12T18:53:53.123Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** AI learns to win (not draw) at Chenapan through AlphaZero-style self-play
**Current focus:** Phase 2 - Pygame GUI

## Current Position

Phase: 2 of 2 (Pygame GUI) — COMPLETE
Plan: 2 of 2 in current phase — COMPLETE
Status: Complete
Last activity: 2026-03-12 — Completed plan 02-02 (AI background thread, checkpoint load, game-over screen, win probability bar)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 12 min
- Total execution time: 47 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-training-foundation | 2 | 37 min | 19 min |
| 02-pygame-gui | 2 | 17 min | 8 min |

**Recent Trend:**
- Last 5 plans: 15 min, 22 min, 7 min, 10 min
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Draw value = -1 (not 0): Forces aggressive play, prevents value head collapse to zero
- pygame-ce over upstream pygame: CE has active development and recent release cadence
- Two-file structure: alpha_pan.py (core) + gui.py (new) — __main__ guard enables safe import
- [Phase 01-training-foundation]: Draw value = -1 (not 0): prevents value head collapse toward zero when draws and losses dominate early training
- [Phase 01-training-foundation]: hashlib.md5 over Python hash(): Python hash() is PYTHONHASHSEED-randomized since 3.3; md5 on tobytes() is stable across runs
- [Phase 01-training-foundation]: Residual architecture over deconv tower: skip connections prevent vanishing gradients, start_block projects 5 channels to 64 hidden
- [Phase 01-training-foundation]: NonWinRate label used instead of DrawRate: outcome -1 is indistinguishable between draw and loss from memory tuple alone
- [Phase 02-pygame-gui]: Single gs dict in main() for mutable game state — avoids Python closure issues with primitives
- [Phase 02-pygame-gui]: SRCALPHA per-surface overlays for highlights — allows semi-transparent yellow/blue cell overlays over board background
- [Phase 02-pygame-gui]: root_value captured at MCTS root's first model call — represents confidence from current board position
- [Phase 02-pygame-gui]: terminal_player flag in gs disambiguates You win / AI wins / Draw in game-over overlay
- [Phase 02-pygame-gui]: Neutral-to-original coord remap: src_orig = 24 - src_idx for rot180 inverse on 25-cell board

### Pending Todos

None yet.

### Blockers/Concerns

- Hyperparameter values (num_iterations, num_selfPlay_iterations) need empirical tuning after first clean run — research recommends 50-200 as a starting range
- Perspective flip correctness in labeling code needs a forced-win trace test (MEDIUM confidence risk flagged in research)

## Session Continuity

Last session: 2026-03-12
Stopped at: Completed 02-pygame-gui-02-PLAN.md — full human-vs-AI loop with AI background thread, checkpoint auto-load, game-over overlay, win probability bar. All phases complete.
Resume file: None

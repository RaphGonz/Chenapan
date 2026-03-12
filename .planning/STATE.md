---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
last_updated: "2026-03-12T18:30:19Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 4
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** AI learns to win (not draw) at Chenapan through AlphaZero-style self-play
**Current focus:** Phase 2 - Pygame GUI

## Current Position

Phase: 2 of 2 (Pygame GUI)
Plan: 1 of 2 in current phase — COMPLETE
Status: In Progress
Last activity: 2026-03-12 — Completed plan 02-01 (gui.py board rendering, click-to-move, side panel)

Progress: [███████░░░] 75%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 19 min
- Total execution time: 37 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-training-foundation | 2 | 37 min | 19 min |
| 02-pygame-gui | 1 | 7 min | 7 min |

**Recent Trend:**
- Last 5 plans: 15 min, 22 min, 7 min
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

### Pending Todos

None yet.

### Blockers/Concerns

- Hyperparameter values (num_iterations, num_selfPlay_iterations) need empirical tuning after first clean run — research recommends 50-200 as a starting range
- Perspective flip correctness in labeling code needs a forced-win trace test (MEDIUM confidence risk flagged in research)

## Session Continuity

Last session: 2026-03-12
Stopped at: Completed 02-pygame-gui-01-PLAN.md — gui.py with board rendering, click-to-move, side panel. Plan 02-01 complete.
Resume file: None

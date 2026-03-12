---
phase: 02-pygame-gui
plan: "02"
subsystem: ui
tags: [pygame, pytorch, threading, mcts, alphazero]

requires:
  - phase: 02-01
    provides: gui.py board rendering, click-to-move, side panel skeleton, GameState enum
  - phase: 01-training-foundation
    provides: AlphaPanNet, MCTS, Chenapan — the trained model and game logic

provides:
  - MCTS.search() returns (action_probs, root_value) 2-tuple
  - AI plays in background thread — window stays responsive at 30 FPS during compute
  - Checkpoint auto-load on startup (model*.pt, most recently modified)
  - Game-over overlay with correct "You win" / "AI wins" / "Draw" determination
  - Win probability bar wired to live MCTS root_value after every AI move
  - Keypress-to-restart without relaunching the program
  - "AI is thinking..." text displayed during background computation

affects: [future-training, deployment]

tech-stack:
  added: [threading, glob, os, torch, numpy — added to gui.py imports]
  patterns:
    - "Background thread + Event flag pattern: thread sets Event on completion; main loop polls is_set() non-blocking"
    - "Neutral-state coordinate remapping: rot180 maps position i -> 24-i on 25-cell board"
    - "gs dict for all mutable game state avoids Python closure issues with primitives"
    - "Checkpoint auto-load by glob + os.path.getmtime — picks most recently saved model.pt"

key-files:
  created: []
  modified:
    - alpha_pan.py
    - gui.py

key-decisions:
  - "root_value captured at MCTS root's first model call (policy init pass) — represents confidence from current position, not from a leaf deep in tree"
  - "terminal_player flag (1=human, -1=AI) stored in gs to disambiguate You win vs AI wins in overlay"
  - "Neutral-to-original coordinate remap: src_orig = 24 - src_idx, dest_orig = 24 - dest_idx (rot180 inverse)"
  - "ai_thread_launched bool flag in gs — cleaner than hasattr check for preventing double-launch"
  - "Font objects for thinking/game-over created outside the event loop to avoid per-frame alloc"

patterns-established:
  - "Non-blocking AI poll: ai_done_event.is_set() checked each frame; never thread.join() in main loop"
  - "Thread worker run_ai() contains zero pygame calls — pure numpy/torch compute"
  - "game.reset() on restart from main thread only — never from AI thread"

requirements-completed: [GUI-03, GUI-04, GUI-05]

duration: 10min
completed: 2026-03-12
---

# Phase 2 Plan 02: AI Integration Summary

**pygame GUI wired to MCTS background thread with checkpoint auto-load, responsive window at 30 FPS, and game-over overlay showing "You win" / "AI wins" / "Draw" with win-probability bar from MCTS root_value**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-12T18:37:39Z
- **Completed:** 2026-03-12T18:49:31Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended MCTS.search() to return (action_probs, root_value) 2-tuple; selfPlay() call sites updated to discard root_value with _
- Full AI integration in gui.py: checkpoint auto-load, background threading, non-blocking event poll, action coordinate remapping, and game-over detection
- Complete game-over overlay with semi-transparent black surface, correct result string based on who moved last, and keypress-to-restart

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend MCTS.search() to return root_value** - `85c33fa` (feat)
2. **Task 2: Wire AI background thread, checkpoint load, complete game-over screen** - `d342cf0` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `alpha_pan.py` - MCTS.search() now captures root_value at root node and returns (action_probs, root_value.item()); selfPlay() player==1 and player==-1 call sites use _ to discard root_value
- `gui.py` - Added load_latest_checkpoint(), run_ai() thread worker, AI thread state machine with ai_done_event polling, terminal_player tracking, complete GAME_OVER overlay, live win bar, "AI is thinking..." text

## Decisions Made

- root_value is captured at the root node's first forward pass (policy initialization), not from leaf nodes — this correctly reflects model confidence at the current board position
- Neutral-to-original coordinate remapping uses rot180 inverse: position i maps to 24-i on the 25-cell board
- terminal_player stored in gs (1=human, -1=AI) to correctly disambiguate the three overlay outcomes
- ai_thread_launched boolean flag used instead of hasattr checks — simpler and resets cleanly on restart

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Model checkpoint (model.pt) is auto-loaded from working directory if present; random weights used otherwise.

## Next Phase Readiness

- Full human-vs-AI loop is now playable: launch `python gui.py`, make a move, AI computes in background, game-over restarts cleanly
- No further GUI phases planned — project milestone v1.0 complete
- Remaining work: hyperparameter tuning of num_iterations / num_selfPlay_iterations after first clean training run

## Self-Check: PASSED

- FOUND: gui.py (488 lines, above 260 minimum)
- FOUND: alpha_pan.py (syntax OK, returns tuple)
- FOUND: 02-02-SUMMARY.md
- FOUND: commit 85c33fa (feat: MCTS root_value)
- FOUND: commit d342cf0 (feat: AI thread integration)

---
*Phase: 02-pygame-gui*
*Completed: 2026-03-12*

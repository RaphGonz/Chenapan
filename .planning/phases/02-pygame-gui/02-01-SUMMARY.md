---
phase: 02-pygame-gui
plan: "01"
subsystem: gui
tags: [pygame, board-rendering, click-to-move, side-panel, game-state]
dependency_graph:
  requires: [alpha_pan.py]
  provides: [gui.py]
  affects: []
tech_stack:
  added: [pygame-ce]
  patterns: [state-dict-closures, srcalpha-overlays, enum-game-state]
key_files:
  created: [gui.py]
  modified: []
decisions:
  - "Single gs dict in main() for mutable game state — avoids Python closure issues with primitives"
  - "Tasks 1 and 2 implemented together in one file creation — both tasks target gui.py with fully interdependent design"
  - "draw_game_over shows 'Draw' for terminal_value=-1 (AI win cannot occur with stub AI)"
metrics:
  duration: "7 min"
  completed: "2026-03-12"
  tasks_completed: 2
  files_created: 1
  files_modified: 0
---

# Phase 02 Plan 01: Pygame GUI Foundation Summary

**One-liner:** Pygame window with 5x5 board rendering, SRCALPHA highlight overlays, click-to-move state machine, and live side panel (move counter, draw tracker, win probability bar).

## What Was Built

`gui.py` implements the human-facing half of the Alpha-Pan game UI. The file imports only `Chenapan`, `MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED`, and `MAX_NUMBER_OF_MOVES` from `alpha_pan` — no training code is touched.

### Key Components

**GameState enum** — `WAITING_FOR_HUMAN`, `AI_THINKING`, `GAME_OVER` — drives the main loop's event routing.

**PIECE_LABEL** — exactly as locked in CONTEXT.md: `{0:"0", 1:"A", 2-9:digits, 10:"V", 11:"D", 12:"R"}`.

**draw_board(screen, state, selected_cell, valid_moves, ai_last_dest):**
- Fills board area with `BOARD_BG` (dark gray)
- Draws 5x5 grid lines in `GRID_COLOR`
- When a piece is selected: SRCALPHA yellow overlay on each valid destination cell
- When `ai_last_dest` is set: SRCALPHA blue overlay on that cell
- For each non-empty cell: draws filled circle (radius 38px) in `RED_PIECE` (positive), `BLACK_PIECE` (negative), or `WHITE_DISK` (joker value 0), then renders `PIECE_LABEL` text centered on disk

**draw_panel(screen, game, ai_value):**
- Renders to the right of the board at `x = BOARD_OFFSET_X + 5*CELL_SIZE + 10`
- "Moves: N" using `game.number_of_moves` (live data)
- "Loops: N / 3" using `game.biggest_loop` and `MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED` (live data)
- Win probability bar: 180px wide, split black (AI share) / red (human share); `ai_value=0.0` renders 50/50 at this stage

**handle_click(pixel_pos, gs, game):**
- Converts pixel coords to board cell via `BOARD_OFFSET_X/Y` and `CELL_SIZE`
- First click on a positive-value cell: sets `gs["selected_cell"]`
- Second click on valid destination: applies move via `game.get_next_state(..., update_meta_parameters=True)`, checks terminal via `game.get_value_and_terminated`, transitions to `GAME_OVER` if terminal or stays `WAITING_FOR_HUMAN` (AI stub no-op)
- Second click on invalid destination: deselects without moving

**draw_game_over(screen, terminal_value):**
- Black SRCALPHA overlay (alpha=160) over full window (board stays visible)
- "You win!" if `terminal_value == 1`, "Draw" otherwise
- "Press any key to restart" prompt

**Restart flow:** `KEYDOWN` in `GAME_OVER` state calls `game.reset()`, reinitializes `gs["state"]`, resets all gs fields.

## Verification Results

All automated checks passed:
- `ast.parse(open('gui.py').read())` — syntax OK
- Module import check — import OK (no pygame display required)
- 373 lines (minimum was 180)
- All required names present: `GameState`, `draw_board`, `draw_panel`, `handle_click`, `main`
- `PIECE_LABEL` dict matches specification exactly

## Deviations from Plan

None — plan executed exactly as written. Both tasks (Task 1 scaffolding + Task 2 handle_click wiring) were implemented in a single file creation since they are fully interdependent and the complete design was specified in the plan. The commit covers both tasks atomically.

## Commits

| Hash    | Message                                                                    |
|---------|----------------------------------------------------------------------------|
| 7ff1aaf | feat(02-01): create gui.py with board rendering, click-to-move, side panel |

## Self-Check: PASSED

- `gui.py` exists: FOUND
- Commit 7ff1aaf exists: FOUND
- 373 lines (>= 180 minimum): PASSED
- `GameState`, `draw_board`, `draw_panel`, `handle_click`, `main` all present: PASSED
- `PIECE_LABEL` exact match: PASSED

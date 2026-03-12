---
phase: 02-pygame-gui
verified: 2026-03-12T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 02: pygame-gui Verification Report

**Phase Goal:** A playable human-vs-AI window where a human plays Chenapan against the strongest available checkpoint
**Verified:** 2026-03-12
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Running `python gui.py` opens a pygame window showing the 5x5 board with grid lines and all pieces visible | VERIFIED | `draw_board()` fills board background, draws 6 vertical + 6 horizontal `pygame.draw.line` calls for grid, iterates `state[row, col]` for all 25 cells drawing circles; `main()` calls `pygame.display.set_mode` and enters render loop at 30 FPS |
| 2  | Player 1 (red) pieces show the correct label mapping: 0="0", 1="A", 2-9=digits, 10="V", 11="D", 12="R" | VERIFIED | `PIECE_LABEL` dict verified via AST: exact match to spec. `draw_board()` uses `PIECE_LABEL[abs(cell_val)]` for positive (red) and negative (black) cells; `RED_PIECE` color applied for `cell_val > 0` |
| 3  | Player 2 (negative) pieces show black disks with the same label mapping on their absolute value | VERIFIED | `draw_board()` else branch (line 197): `pygame.draw.circle(screen, BLACK_PIECE, ...)`, `PIECE_LABEL[abs(cell_val)]` — confirmed identical mapping path |
| 4  | Clicking a player-1 piece highlights all valid destination squares with a colored overlay | VERIFIED | `handle_click()` sets `gs["selected_cell"]` for `cell_val > 0`; `draw_board()` reads `valid_moves[src_idx]` and calls `_draw_highlight` with `HIGHLIGHT_RGBA` (yellow, alpha=100, SRCALPHA surface) for each dest; `valid_moves` computed via `game.get_valid_moves(gs["state"], 1)` every frame |
| 5  | Clicking anywhere that is not a valid destination deselects the piece without moving it | VERIFIED | `handle_click()` else branch (line 321-323): `gs["selected_cell"] = None` — no move applied, no state change |
| 6  | The side panel shows: move counter, draw loop tracker (X / 3 loops), and win probability bar; bar updates with live MCTS value after each AI move | VERIFIED | `draw_panel()` renders `f"Moves: {game.number_of_moves}"`, `f"Loops: {game.biggest_loop} / {MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED}"`, and a bar with `ai_share = (ai_value + 1.0) / 2.0`; `gs["ai_value"]` is written by `run_ai()` from `root_value` and read by `draw_panel(screen, game, gs["ai_value"])` each frame |
| 7  | After the human moves, the window remains responsive while the AI computes; AI move destination is highlighted; game-over overlay shows correct result; key press restarts | VERIFIED | `threading.Thread(target=run_ai, ..., daemon=True)` launched in main loop body; main loop polls `ai_done_event.is_set()` non-blocking — no `thread.join()`; `gs["ai_last_dest"]` set after AI move for blue highlight; GAME_OVER overlay renders "You win"/"AI wins"/"Draw" via `terminal_player` flag; KEYDOWN in GAME_OVER calls `game.reset()` and reinitializes all `gs` fields |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `gui.py` | 260 (plan 02) | 488 | VERIFIED | Complete implementation: `GameState`, `load_latest_checkpoint`, `run_ai`, `draw_board`, `draw_panel`, `handle_click`, `main`. Syntax OK. |
| `alpha_pan.py` | — | 952 | VERIFIED | `MCTS.search()` returns `(action_probs, root_value.item())` tuple (line 712). `policy, root_value = self.model(...)` at line 649. `selfPlay()` call sites updated to `action_probs, _ = self.mcts.search(...)` at lines 735 and 749. Syntax OK. Live test confirmed 2-tuple return. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `gui.py click handler` | `Chenapan.get_valid_moves(state, 1)` | `valid_moves` list-of-list indexed by `src_idx` | WIRED | Line 302: `valid_moves = game.get_valid_moves(state, 1)` in `handle_click`; line 440: computed again each frame for rendering highlights |
| `gui.py draw_board()` | `state numpy array` | `state[row, col]` read for piece color and label | WIRED | Lines 179, 290: `cell_val = int(state[row, col])` — both in draw_board and handle_click |
| `gui.py AI thread function` | `mcts.search(neutral_state_copy)` | `threading.Thread(target=run_ai, ...)`; result stored in `gs` dict under `ai_lock` | WIRED | Line 404: `t = threading.Thread(target=run_ai, args=(neutral_state_copy, mcts, gs, ai_lock, ai_done_event), daemon=True)`. `run_ai()` calls `mcts.search()` at line 102, stores result in `gs["ai_action"]` and `gs["ai_value"]` under `ai_lock` |
| `gui.py main loop` | `ai_done_event.is_set()` | Non-blocking poll — never `thread.join()` in main loop | WIRED | Line 410: `elif ai_done_event.is_set():` inside the `AI_THINKING` block; confirmed no `thread.join()` call anywhere in file |
| `gui.py draw_panel` | `gs["ai_value"]` | `bar_pct = (ai_value + 1) / 2`; draw rect proportional to `bar_pct` | WIRED | Line 449: `draw_panel(screen, game, gs["ai_value"])`; lines 247-258: `ai_share = (ai_value + 1.0) / 2.0` used to split bar |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GUI-01 | 02-01 | pygame-ce window renders 5x5 board with piece values visible for both players | SATISFIED | `draw_board()` renders all 25 cells; piece circles with labels for positive (red), negative (black), and 0 (white/joker) |
| GUI-02 | 02-01 | Human player selects a piece then clicks destination to move; valid destinations highlighted | SATISFIED | `handle_click()` implements select-then-move; `draw_board()` renders yellow SRCALPHA highlights on `valid_moves[src_idx]` |
| GUI-03 | 02-02 | AI move computed in background thread; no window freeze during MCTS search | SATISFIED | `threading.Thread(target=run_ai, daemon=True)`; main loop polls `ai_done_event.is_set()` non-blocking; `run_ai()` has zero pygame calls |
| GUI-04 | 02-02 | Game-over screen displays outcome (win/loss/draw); pressing a key restarts without relaunching | SATISFIED | GAME_OVER overlay with "You win"/"AI wins"/"Draw" logic using `terminal_player` flag; KEYDOWN calls `game.reset()` and reinitializes all `gs` fields |
| GUI-05 | 02-02 | Latest model checkpoint auto-detected and loaded on startup with `model.eval()` | SATISFIED | `load_latest_checkpoint()` globs `model*.pt`, picks most recently modified via `os.path.getmtime`, calls `model.eval()` in both branches (checkpoint found and not found) |

All 5 phase-2 requirement IDs accounted for. No orphaned requirements.

---

### Anti-Patterns Found

No anti-patterns found.

| File | Line | Pattern | Severity | Notes |
|------|------|---------|----------|-------|
| — | — | — | — | No TODO/FIXME/placeholder comments, no empty returns, no stub implementations |

Additional checks:
- `run_ai()` contains zero pygame calls (confirmed by body scan)
- No `thread.join()` anywhere in `gui.py`
- No `tqdm` import in `gui.py`
- `font_big` and `font_small` are created inside the game-over render block per frame — minor per-frame allocation, but not a functional correctness issue

---

### Human Verification Required

The following behaviors are correct in code but require a human to confirm end-to-end during an actual run:

#### 1. Window Responsiveness During AI Computation

**Test:** Make a human move. Observe the window while the AI computes (num_searches=60 may take several seconds on CPU).
**Expected:** Board stays visible and redraws continuously; "AI is thinking..." text appears; window does not show "(Not Responding)" in title bar.
**Why human:** Threading behavior cannot be verified statically.

#### 2. AI Move Highlight

**Test:** After the AI completes its move, verify the destination cell shows a blue highlight overlay.
**Expected:** The cell where the AI piece landed is highlighted in `AI_LAST_MOVE_RGBA` (light blue) until the human makes their next move.
**Why human:** Requires runtime rendering to confirm visually.

#### 3. Win Probability Bar Live Update

**Test:** Play several moves. Observe the side panel win probability bar after each AI move.
**Expected:** The bar changes its split proportions reflecting the AI's MCTS root_value, not staying fixed at 50/50 after the first move.
**Why human:** Requires observing dynamic value changes during gameplay.

#### 4. Game-Over Correct Outcome String

**Test:** Play to a terminal state (both "You win" by capturing all opponent pieces, and a draw scenario if possible).
**Expected:** "You win", "AI wins", or "Draw" shown correctly based on who triggered the terminal condition.
**Why human:** Terminal condition paths depend on actual game outcomes that can only be reached during play.

---

### Gaps Summary

No gaps found. All automated checks passed.

---

## Summary

Phase 02 goal is fully achieved. The codebase delivers a playable human-vs-AI pygame window:

- `gui.py` (488 lines) implements the complete rendering pipeline, click-to-move state machine, AI background thread with non-blocking event poll, checkpoint auto-load with `model.eval()`, and game-over overlay with correct outcome determination.
- `alpha_pan.py` has been correctly extended: `MCTS.search()` returns `(action_probs, root_value.item())` and both `selfPlay()` call sites discard `root_value` with `_` unpacking.
- All 5 requirement IDs (GUI-01 through GUI-05) are satisfied with concrete implementation evidence.
- No stubs, placeholders, or anti-patterns detected.
- Four items flagged for human smoke-test verification (window responsiveness, AI highlight rendering, live win bar, and game-over strings) — these are runtime behaviors that pass static analysis.

---

_Verified: 2026-03-12_
_Verifier: Claude (gsd-verifier)_

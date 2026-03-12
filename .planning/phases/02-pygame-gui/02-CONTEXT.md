# Phase 2: Pygame GUI - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

A playable human-vs-AI pygame window for Chenapan. Covers board rendering, piece display, click-to-move with valid move highlighting, non-blocking AI computation, checkpoint auto-load, game-over screen with restart, and a side panel with game state info. Creating a new game engine or changing game rules is out of scope.

</domain>

<decisions>
## Implementation Decisions

### Board & Piece Appearance
- Plain grid — no alternating square colors, just grid lines separating cells
- Clean and minimal visual style — no decorative elements, functional first
- Each piece is rendered as a disk with a label on top
- Piece notation:
  - 0 → white disk, labeled "0"
  - 1 → "A"
  - 2–9 → their numeric value
  - 10 → "V"
  - 11 → "D"
  - 12 → "R"
- Positive pieces (player 1) → red disks
- Negative pieces (player 2) → black disks

### Move Selection UX
- Clicking a piece highlights all valid destination squares with a colored overlay
- Clicking anywhere that is not a valid destination deselects the piece (no move made)
- Human always plays as player 1 (red, moves first)

### AI Turn Feedback
- While AI is computing: display "AI is thinking..." text on screen (board remains visible and responsive — no freeze)
- After the AI plays: highlight the destination square of its last move so the human can clearly see where it moved

### Game-Over Screen
- Semi-transparent overlay on top of the final board position (board stays visible)
- Shows result: "You win", "AI wins", or "Draw"
- Shows "Press any key to restart" prompt
- Pressing any key resets the game without relaunching the program

### Side Panel
- Displayed to the right of (or below) the board
- Contains:
  - **Move counter** — total moves made in the current game
  - **Draw loop tracker** — current repeated-position count vs. the allowed maximum (e.g. "3 / 5 loops") — uses the existing hash-based draw detection from Phase 1
  - **Win probability bar** — displays the AI's value estimate as a percentage bar (like a chess engine eval bar); updates after each AI move using the value the AI already computed (no extra model calls on human turns)

### Claude's Discretion
- Exact colors for board, highlighting, and overlay (as long as they are readable and consistent with the red/black piece scheme)
- Font choice and text sizing
- Window dimensions and board cell sizing
- How the side panel is laid out internally

</decisions>

<specifics>
## Specific Ideas

- Piece notation is custom and fixed: 0=white zero, 1=A, 2–9=numbers, 10=V, 11=D, 12=R — the planner must use exactly this mapping
- Win bar should feel like a standard chess engine eval bar — a horizontal or vertical bar split between the two sides, updating each turn after the AI plays
- Draw loop tracker should surface the internal repetition counter that already exists in `Chenapan` (check `get_value_and_terminated` and hash tracking in `alpha_pan.py`)

</specifics>

<deferred>
## Deferred Ideas

- Showing win probability after every human move (requires extra background model call per human turn) — could be added in a later polish phase
- Any form of move history / PGN log

</deferred>

---

*Phase: 02-pygame-gui*
*Context gathered: 2026-03-12*

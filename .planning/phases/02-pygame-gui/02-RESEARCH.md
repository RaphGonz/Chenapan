# Phase 2: Pygame GUI - Research

**Researched:** 2026-03-12
**Domain:** pygame-ce game loop, threading, board rendering, PyTorch inference
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Board & Piece Appearance
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

#### Move Selection UX
- Clicking a piece highlights all valid destination squares with a colored overlay
- Clicking anywhere that is not a valid destination deselects the piece (no move made)
- Human always plays as player 1 (red, moves first)

#### AI Turn Feedback
- While AI is computing: display "AI is thinking..." text on screen (board remains visible and responsive — no freeze)
- After the AI plays: highlight the destination square of its last move so the human can clearly see where it moved

#### Game-Over Screen
- Semi-transparent overlay on top of the final board position (board stays visible)
- Shows result: "You win", "AI wins", or "Draw"
- Shows "Press any key to restart" prompt
- Pressing any key resets the game without relaunching the program

#### Side Panel
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

### Deferred Ideas (OUT OF SCOPE)
- Showing win probability after every human move (requires extra background model call per human turn) — could be added in a later polish phase
- Any form of move history / PGN log
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GUI-01 | pygame-ce window renders 5×5 board with piece values visible for both players | Standard pygame-ce Surface/draw API; piece notation mapping documented in CONTEXT.md |
| GUI-02 | Human player selects a piece then clicks destination to move; valid destinations highlighted after piece selection | `pygame.MOUSEBUTTONDOWN` event + `get_valid_moves()` from `Chenapan`; highlight via SRCALPHA surface blit |
| GUI-03 | AI move computed in background thread — prevents window freeze during MCTS search | `threading.Thread` + shared state with `threading.Lock`; main thread runs event loop and rendering exclusively |
| GUI-04 | Game-over screen displays outcome (win/loss/draw); pressing a key restarts without relaunching | SRCALPHA overlay surface; `KEYDOWN` event resets game state; `Chenapan.reset()` exists |
| GUI-05 | Latest model checkpoint auto-detected and loaded on startup with `model.eval()` | `glob.glob("model*.pt")` + `max(..., key=os.path.getmtime)`; current code saves to `model.pt` (single file); `torch.load` + `model.eval()` |
</phase_requirements>

---

## Summary

Phase 2 builds `gui.py` — a new standalone file that imports `Chenapan`, `AlphaPanNet`, and `MCTS` from `alpha_pan.py` (protected by `__main__` guard from Phase 1). The core challenge is keeping the pygame window responsive while MCTS searches, which requires the AI computation to run in a background `threading.Thread` with the main thread retaining exclusive control over all pygame API calls.

The current codebase saves checkpoints as `model.pt` (a single fixed-name file, overwritten each iteration). Auto-detection is therefore trivial: check for `model.pt` in the working directory. If the training pipeline is later extended to save numbered checkpoints, `glob` + `max(key=os.path.getmtime)` handles both cases. The GUI reads `biggest_loop` and `number_of_moves` directly from the `Chenapan` instance already in scope — no new state tracking is needed.

The side panel's win probability bar uses the value returned from the MCTS `search()` call (which also returns `value` from the root's value head). The MCTS class needs a one-line extension to surface that value. All rendering, event handling, and surface blitting must remain on the main thread; only the `mcts.search()` call crosses into the background thread.

**Primary recommendation:** Single `gui.py` file, game-loop state machine (WAITING_FOR_HUMAN / AI_THINKING / GAME_OVER), background thread for MCTS only, all pygame calls on main thread, shared result communicated via a threading.Event + plain Python attribute protected by a Lock.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pygame-ce | 2.5.7 | Window, rendering, event loop | Locked decision from STATE.md; active development, latest release March 2026 |
| threading | stdlib | Background AI computation | Sufficient for GIL-released PyTorch CPU/GPU calls; no extra install |
| torch | project's existing | Load checkpoint, run inference | Already in project; `model.eval()` and `torch.no_grad()` for inference |
| glob + os | stdlib | Auto-detect latest checkpoint | Matches `model*.pt` pattern; `os.path.getmtime` for newest-by-time |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | project's existing | Board state arrays passed to MCTS | Already used throughout alpha_pan.py |
| sys | stdlib | Clean exit on window close | `sys.exit()` after `pygame.quit()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| threading | multiprocessing | multiprocessing avoids GIL but has serialization overhead for numpy arrays; threading is sufficient since PyTorch releases GIL during GPU/CPU ops |
| threading | asyncio | asyncio is for I/O-bound concurrency; MCTS is CPU/GPU-bound; threading is correct here |
| plain pygame-ce | pygame (upstream) | STATE.md locked this decision; pygame-ce has more active releases |

**Installation:**
```bash
pip install pygame-ce
# If upstream pygame is already installed, uninstall first:
# pip uninstall pygame
```

---

## Architecture Patterns

### Recommended Project Structure
```
Alpha-Pan/
├── alpha_pan.py        # Existing: Chenapan, AlphaPanNet, MCTS, AlphaPan (unchanged)
├── gui.py              # New: entire GUI implementation
└── model.pt            # Written by training; read by gui.py on startup
```

`gui.py` is a self-contained file. It imports only from `alpha_pan` and standard library.

### Pattern 1: State Machine Game Loop

**What:** Main loop runs a fixed-FPS `while True` that dispatches on a `game_state` enum: `WAITING_FOR_HUMAN`, `AI_THINKING`, `GAME_OVER`. Each state has its own event handling and drawing path.

**When to use:** Any pygame game with distinct phases where input behavior differs between phases.

```python
# Source: standard pygame pattern
import pygame
from enum import Enum, auto

class GameState(Enum):
    WAITING_FOR_HUMAN = auto()
    AI_THINKING = auto()
    GAME_OVER = auto()

game_state = GameState.WAITING_FOR_HUMAN
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if game_state == GameState.WAITING_FOR_HUMAN:
            handle_human_input(event)
        elif game_state == GameState.GAME_OVER:
            if event.type == pygame.KEYDOWN:
                restart_game()

    draw_board()
    if game_state == GameState.AI_THINKING:
        draw_thinking_text()
    if game_state == GameState.GAME_OVER:
        draw_overlay()

    pygame.display.flip()
    clock.tick(30)
```

### Pattern 2: Background Thread for AI with Lock-Protected Result

**What:** Spawn a `threading.Thread` to run `mcts.search()`. Use a `threading.Lock` to protect the shared result slot. Main thread polls a `threading.Event` (non-blocking `.is_set()`) to know when the AI is done.

**When to use:** Any compute-bound operation that must not block the main pygame event loop (required for GUI-03).

**Critical rule:** Never call any `pygame.*` function from inside the background thread. Only read/write plain Python data structures and torch tensors there.

```python
# Source: standard Python threading pattern
import threading

ai_result = {"action": None, "value": None}
ai_done_event = threading.Event()
ai_lock = threading.Lock()

def run_ai(game, mcts, state):
    # All MCTS computation happens here — no pygame calls
    action_probs = mcts.search(state)
    action = np.unravel_index(np.argmax(action_probs), action_probs.shape)
    # Capture value from MCTS root for win bar (see Pattern 4)
    with ai_lock:
        ai_result["action"] = action
    ai_done_event.set()

# When entering AI_THINKING state:
ai_done_event.clear()
t = threading.Thread(target=run_ai, args=(game, mcts, state_copy), daemon=True)
t.start()

# In main loop, while AI_THINKING:
if ai_done_event.is_set():
    with ai_lock:
        action = ai_result["action"]
    apply_ai_move(action)
    game_state = GameState.WAITING_FOR_HUMAN
```

**Important:** Pass a `state.copy()` to the thread to avoid race conditions on the numpy array.

### Pattern 3: Semi-Transparent Game-Over Overlay

**What:** Create a `pygame.Surface` with `SRCALPHA` flag, fill it with a semi-transparent dark color, blit it over the board, then draw result text on top.

**When to use:** Game-over screen that keeps the board visible (required by locked decision).

```python
# Source: pygame-ce Surface docs (https://pyga.me/docs/ref/surface.html)
overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
overlay.fill((0, 0, 0, 160))  # RGBA — 160/255 opacity
screen.blit(overlay, (0, 0))

font = pygame.font.SysFont(None, 64)
text = font.render("You win", True, (255, 255, 255))
screen.blit(text, text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))
```

### Pattern 4: Value Head Win Probability Bar

**What:** The MCTS `search()` method already calls `self.model(...)` which returns `(policy, value)`. The value at the root is the AI's current confidence. Surface this for the win bar by storing the root's first value call result.

**Current code gap:** `MCTS.search()` returns `action_probs` (visit counts) but discards the root's value estimate. A minimal addition is needed: store `value` from the first model call (line 651 in alpha_pan.py) and return it alongside `action_probs`.

**Minimal change to MCTS.search():**
```python
# In MCTS.search(), after line: policy,_ = self.model(...)
# Change to:
policy, root_value = self.model(
    torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
)
# ... existing code ...
# At return, yield root_value.item() alongside action_probs
return action_probs, root_value.item()
```

The GUI then converts this scalar (in [-1, 1], where 1 = current player wins) to a bar percentage: `bar_pct = (root_value + 1) / 2` — 0% = AI certain to lose, 100% = AI certain to win.

### Pattern 5: Checkpoint Auto-Detection

**What:** On startup, find the most recent `.pt` file matching the training output pattern.

**Current state:** `alpha_pan.py` saves `model.pt` only (single fixed file). Auto-detection is a one-liner for now; structured to survive future numbered checkpoints.

```python
# Source: Python stdlib glob + os
import glob, os

def load_latest_checkpoint(model, device):
    candidates = glob.glob("model*.pt")
    if not candidates:
        print("No checkpoint found — model uses random weights")
        model.eval()
        return False
    latest = max(candidates, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {latest}")
    return True
```

### Pattern 6: Valid Move Highlighting

**What:** After a piece is selected, compute `game.get_valid_moves(state, 1)` and draw a colored overlay on each valid destination cell before rendering pieces.

```python
selected_cell = None  # (row, col) or None

def handle_click(pixel_x, pixel_y):
    global selected_cell, game_state
    col = (pixel_x - BOARD_OFFSET_X) // CELL_SIZE
    row = (pixel_y - BOARD_OFFSET_Y) // CELL_SIZE
    if not (0 <= row < 5 and 0 <= col < 5):
        selected_cell = None
        return

    cell_idx = row * 5 + col
    valid_moves = game.get_valid_moves(state, 1)

    if selected_cell is None:
        # Select if own piece
        if state[row, col] > 0:
            selected_cell = (row, col)
    else:
        src_idx = selected_cell[0] * 5 + selected_cell[1]
        if cell_idx in valid_moves[src_idx]:
            apply_human_move([src_idx, cell_idx])
            selected_cell = None
        else:
            selected_cell = None  # deselect, no move

def draw_highlights(screen, valid_moves, src_idx):
    highlight = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    highlight.fill((255, 255, 0, 100))  # yellow, semi-transparent
    for dest_idx in valid_moves[src_idx]:
        r, c = dest_idx // 5, dest_idx % 5
        screen.blit(highlight, (BOARD_OFFSET_X + c * CELL_SIZE, BOARD_OFFSET_Y + r * CELL_SIZE))
```

### Anti-Patterns to Avoid

- **Calling `pygame.draw.*` or `screen.blit()` from the background AI thread:** SDL is not thread-safe for rendering calls; will crash or corrupt display on Windows.
- **Passing `state` (mutable numpy array) directly to the AI thread without `.copy()`:** MCTS internally calls `state.copy()` in `expand()`, but the top-level state can be mutated by `get_next_state` in `simulate()` — always pass `state.copy()` to the thread.
- **Blocking on `thread.join()` in the main loop:** This freezes the window. Use `ai_done_event.is_set()` (non-blocking poll) instead.
- **Calling `game.reset()` inside the AI thread:** `Chenapan.reset()` modifies instance state (`number_of_moves`, `biggest_loop`, `list_of_positions`) — only call from main thread.
- **Reading `game.biggest_loop` for the draw tracker while AI thread is running:** The AI thread may be calling `get_next_state` with `update_meta_parameters=False`, but protect reads from the main thread with the lock anyway.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Semi-transparent overlay | Custom pixel blending loop | `pygame.Surface` with `SRCALPHA` + `fill((r,g,b,a))` | Built-in per-pixel alpha compositing, one blit call |
| Font rendering | PIL/custom text | `pygame.font.SysFont` or `pygame.font.Font` | pygame-ce has full font rendering; no extra library |
| Thread-safe result passing | Global variable with no lock | `threading.Lock` + `threading.Event` | Race conditions are real; Lock prevents torn reads |
| Checkpoint discovery | Manual filename hardcoding | `glob.glob` + `os.path.getmtime` | Handles any future naming convention without code changes |
| Circle/disk drawing | Polygon math | `pygame.draw.circle` | Built into pygame, handles antialiasing with `pygame.draw.aacircle` in pygame-ce |

**Key insight:** pygame-ce's drawing primitives (`draw.circle`, `draw.rect`, `draw.line`) and Surface alpha compositing are the entire rendering toolkit needed for this phase — no external graphics library is required.

---

## Common Pitfalls

### Pitfall 1: AI Thread Freezes Window
**What goes wrong:** `mcts.search()` called synchronously in the main loop blocks event processing; window shows "(Not Responding)" on Windows while MCTS runs.
**Why it happens:** Python's event loop and pygame both run single-threaded; a 60-search MCTS pass takes seconds.
**How to avoid:** Always spawn MCTS as a `threading.Thread`; never call `mcts.search()` directly in the main game loop.
**Warning signs:** Window becomes non-responsive for several seconds after human moves.

### Pitfall 2: Calling pygame Rendering from Background Thread
**What goes wrong:** Intermittent crashes, black screen, or SDL assertion errors on Windows.
**Why it happens:** SDL (pygame's backend) is not thread-safe for video functions. The main thread must own the display.
**How to avoid:** Background thread only touches numpy arrays and torch tensors. All `screen.blit`, `pygame.draw.*`, `pygame.display.flip` stay in main thread.
**Warning signs:** Works on one platform, crashes on another.

### Pitfall 3: Mutable State Shared with AI Thread
**What goes wrong:** Board state gets corrupted mid-MCTS, moves appear to teleport pieces.
**Why it happens:** `get_next_state` modifies the array in-place when called inside `MCTS.search()` with `update_meta_parameters=False` — but the top-level state reference is the same object.
**How to avoid:** Always pass `state.copy()` to the thread function. The `Chenapan` instance itself must also be thread-local or protected — safest is to pass copies of relevant state into the thread and not share the live `Chenapan` instance.
**Warning signs:** Board shows illegal positions or pieces in wrong locations after AI moves.

### Pitfall 4: Model in Training Mode During Inference
**What goes wrong:** BatchNorm uses batch statistics instead of running statistics; Dropout randomly zeroes activations — AI plays inconsistently or badly.
**Why it happens:** Forgetting `model.eval()` after loading checkpoint.
**How to avoid:** Always call `model.eval()` immediately after `model.load_state_dict(...)`. Wrap inference in `torch.no_grad()` context.
**Warning signs:** AI makes different moves when called with identical board positions.

### Pitfall 5: `Chenapan` State Not Reset Between Games
**What goes wrong:** `number_of_moves` and `biggest_loop` carry over from previous game; draw detection triggers immediately on restart.
**Why it happens:** `Chenapan.reset()` is not called when restarting.
**How to avoid:** Call `game.reset()` and re-initialize `state = game.get_initial_state()` on every restart. The `reset()` method exists and clears all three fields.
**Warning signs:** Games end in draw after very few moves on replay.

### Pitfall 6: Perspective Flip on Human Move
**What goes wrong:** Human move applied from wrong perspective; pieces swap in unexpected ways.
**Why it happens:** In self-play, player -1 always sees a `change_perspective(state)` view. In GUI, the human is always player 1 and sees the raw state. The AI must receive `change_perspective(state)` as input to MCTS, and its returned action must be applied to the `change_perspective` view then flipped back — exactly as the commented-out code at the bottom of `alpha_pan.py` shows.
**How to avoid:** Follow the exact pattern from the commented-out human-vs-AI code at the bottom of `alpha_pan.py` (lines ~894–951): human acts on `state` directly; AI acts on `neutral_state = game.change_perspective(state)`, applies action to `neutral_state`, then `state = game.change_perspective(neutral_state)`.
**Warning signs:** Board looks mirrored or AI pieces move as if controlled by player 1.

---

## Code Examples

Verified patterns from official sources:

### pygame-ce Circle Drawing (for piece disks)
```python
# Source: https://pyga.me/docs/ref/draw.html
import pygame

# Filled circle (piece body)
pygame.draw.circle(surface, color=(220, 50, 50), center=(cx, cy), radius=radius)

# Anti-aliased circle outline (pygame-ce has draw.aacircle)
# For pygame-ce, use draw.circle with width=0 for filled, width>0 for outline
pygame.draw.circle(surface, color=(0, 0, 0), center=(cx, cy), radius=radius, width=2)
```

### System Font Text Rendering
```python
# Source: https://pyga.me/docs/ref/font.html
pygame.font.init()
font_large = pygame.font.SysFont("Arial", 28)
font_small = pygame.font.SysFont("Arial", 18)

# Render centered on cell
text_surf = font_large.render("A", True, (255, 255, 255))  # white label on red disk
text_rect = text_surf.get_rect(center=(cx, cy))
surface.blit(text_surf, text_rect)
```

### PyTorch Inference (no_grad)
```python
# Source: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
model.eval()
with torch.no_grad():
    encoded = torch.tensor(
        game.get_encoded_state(state), dtype=torch.float32, device=device
    ).unsqueeze(0)
    policy, value = model(encoded)
```

### Piece Notation Mapping
```python
# Source: CONTEXT.md locked decisions
PIECE_LABEL = {
    0: "0",
    1: "A", 2: "2", 3: "3", 4: "4", 5: "5",
    6: "6", 7: "7", 8: "8", 9: "9",
    10: "V", 11: "D", 12: "R"
}
# Usage: label = PIECE_LABEL[abs(piece_value)]
# Color: red if piece_value > 0, black if piece_value < 0, white if piece_value == 0
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| upstream `pygame` | `pygame-ce` 2.5.7 | STATE.md decision | Active maintenance, SDL3 prep underway |
| `pygame` (upstream) single-threaded AI | `threading.Thread` for MCTS | Required for GUI-03 | Keeps window responsive during search |

**Deprecated/outdated:**
- `tqdm.notebook`: Already replaced in Phase 1 with plain `tqdm` — do not reintroduce in gui.py
- `pygame.font.init()` before font use: Still required in pygame-ce; do not skip

---

## Open Questions

1. **MCTS.search() does not currently return the root value**
   - What we know: The first `model(...)` call on line 651 captures `_` (discarded value). The win bar needs this.
   - What's unclear: Whether to modify `MCTS.search()` to return `(action_probs, value)` or capture value separately in gui.py by calling the model directly.
   - Recommendation: Modify `MCTS.search()` to return `(action_probs, root_value)` — minimal two-character change to line 651 (`policy,_` → `policy,root_value`) and one change to the return statement. This is cleaner than a second model call in gui.py.

2. **Checkpoint naming: single file vs. numbered files**
   - What we know: Current `learn()` saves only `model.pt` (overwritten each iteration). No iteration number in filename.
   - What's unclear: Whether training will be run long enough to produce multiple checkpoints with different names before GUI is used.
   - Recommendation: `glob.glob("model*.pt")` covers both `model.pt` and any future `model_iter_050.pt`. If only `model.pt` exists, `max()` returns it trivially.

3. **Thread safety of reading `game.biggest_loop` and `game.number_of_moves`**
   - What we know: AI thread calls `get_next_state` with `update_meta_parameters=False` (does not touch these fields). The `Chenapan` instance used in MCTS is the same one used in the main game.
   - What's unclear: Whether the MCTS `simulate()` method (not used in AlphaZero path) could ever be triggered.
   - Recommendation: Since MCTS is AlphaZero-style (no rollout/simulate called), and `expand` uses `update_meta_parameters=False`, reading `biggest_loop` and `number_of_moves` from the main thread while AI runs is safe. Document this assumption in code comments.

---

## Sources

### Primary (HIGH confidence)
- pygame-ce PyPI page (https://pypi.org/project/pygame-ce/) — version 2.5.7, released 2026-03-02
- pygame-ce Surface docs (https://pyga.me/docs/ref/surface.html) — SRCALPHA, set_alpha, blit patterns
- PyTorch tutorials (https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — model.eval(), load_state_dict
- `alpha_pan.py` (project source) — Chenapan API, MCTS interface, checkpoint save pattern

### Secondary (MEDIUM confidence)
- pygame-ce GitHub releases (https://github.com/pygame-community/pygame-ce/releases) — version cadence verified
- Standard Python threading docs (stdlib) — threading.Thread, threading.Lock, threading.Event patterns
- pygame draw docs (https://pyga.me/docs/ref/draw.html) — circle, rect, line primitives

### Tertiary (LOW confidence)
- Community GameDev.net thread on pygame threading (https://www.gamedev.net/forums/topic/704486-python-pygame-threading-sample/) — threading pattern validated against stdlib docs
- Glyph blog on pygame mainloop (https://glyph.twistedmatrix.com/2022/02/a-better-pygame-mainloop.html) — mainloop best practices, cross-referenced with pygame docs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pygame-ce version confirmed from PyPI, threading is stdlib
- Architecture: HIGH — patterns verified against pygame-ce docs and project source code
- Pitfalls: HIGH — perspective flip, reset, and thread safety derived directly from reading alpha_pan.py source

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (pygame-ce is stable; threading patterns are stdlib)

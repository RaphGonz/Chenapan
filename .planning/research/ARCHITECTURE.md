# Architecture Research

**Domain:** AlphaZero self-play RL system with pygame board game GUI
**Researched:** 2026-03-12
**Confidence:** HIGH (based on direct source code analysis + verified reference implementations)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Execution Modes                          │
├──────────────────────┬──────────────────────────────────────┤
│   Training Mode      │          Play Mode                   │
│  (alpha_pan.py)      │        (gui.py — new)                │
├──────────────────────┼──────────────────────────────────────┤
│                 Core Domain (alpha_pan.py)                   │
├─────────────┬────────────────┬────────────────┬────────────┤
│  Chenapan   │  AlphaPanNet   │   MCTS + Node  │  AlphaPan  │
│ (game eng.) │ (policy+value) │  (tree search) │ (trainer)  │
└─────────────┴────────────────┴────────────────┴────────────┘
                         │
              ┌──────────┴──────────┐
              │    Shared State     │
              │   NumPy board       │
              │   PyTorch model     │
              │   model_N.pt files  │
              └─────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `Chenapan` | Game rules, move validation, state transitions, draw conditions | `alpha_pan.py` lines 84–390 |
| `get_encoded_state` | Converts NumPy board → 5-channel float32 tensor for neural net | `alpha_pan.py` lines 391–414 |
| `AlphaPanNet` | CNN dual-head: policy (25×25 probs) + value (scalar -1..1) | `alpha_pan.py` lines 416–494 |
| `Node` | MCTS tree node: UCB selection, visit counts, backpropagation | `alpha_pan.py` lines 496–649 |
| `MCTS` | Orchestrates search: expand root, run N simulations, return action probs | `alpha_pan.py` lines 651–726 |
| `AlphaPan` | Training loop: self-play → memory → batch SGD → checkpoint | `alpha_pan.py` lines 728–837 |
| `gui.py` (new) | pygame window, board rendering, click-to-move, AI turn dispatch | new file |
| `config` (inline `args`) | Hyperparameters dict (C, num_searches, batch_size, temperature, etc.) | `alpha_pan.py` lines 844–854 |

## Recommended Project Structure

The decision here is **keep core logic in one file, add GUI as a second file**.

```
Alpha-Pan/
├── alpha_pan.py        # Unchanged core: Chenapan, AlphaPanNet, Node, MCTS, AlphaPan
│                       # training entry point at bottom (guarded by __main__ check)
├── gui.py              # New: pygame GUI — imports from alpha_pan.py
├── model_N.pt          # Trained model checkpoints (existing)
├── optim_N.pt          # Optimizer checkpoints (existing)
└── .planning/          # GSD documentation
```

**Why not split into many files?** The project is ~930 lines and all components are tightly coupled (MCTS takes `game` + `model`, `AlphaPan` takes all four). Splitting into `game.py`, `model.py`, `mcts.py`, `trainer.py` adds import complexity and circular dependency risk without benefit at this scale. The `gui.py` separation is justified because GUI has a distinct execution context (no training) and a different event loop (pygame).

**Why add `gui.py` as a separate file?** The training loop (`alphaPan.learn()`) and the pygame event loop are mutually exclusive execution paths. Mixing them in one file would require conditional top-level execution that is harder to read and harder to extend.

### Structure Rationale

- **`alpha_pan.py`:** All RL logic. The bottom script block becomes `if __name__ == "__main__": alphaPan.learn()` so the module can be imported by `gui.py` without triggering training.
- **`gui.py`:** Imports `Chenapan`, `AlphaPanNet`, `MCTS`, `args` from `alpha_pan`. Loads the latest `model_N.pt` at startup. Runs a pygame event loop where human inputs moves by clicking and the AI responds via MCTS.

## Architectural Patterns

### Pattern 1: Guard Training Entry Point with `__main__`

**What:** Wrap the bottom of `alpha_pan.py` in `if __name__ == "__main__":` so that `import alpha_pan` in `gui.py` does not start training.

**When to use:** Any time a Python module serves both as a runnable script and as an importable library.

**Trade-offs:** Zero cost. Required for the two-file architecture to work at all.

**Example:**
```python
# Bottom of alpha_pan.py — was bare module-level code:
if __name__ == "__main__":
    game = Chenapan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPanNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = { ... }
    alphaPan = AlphaPan(model, optimizer, game, args)
    alphaPan.learn()
```

### Pattern 2: Load Trained Model for Inference

**What:** In `gui.py`, load the highest-numbered checkpoint, set `model.eval()`, and pass it to an `MCTS` instance. The game engine (`Chenapan`) and the `args` dict are re-created identically to training conditions.

**When to use:** Any play mode where the user faces the trained AI.

**Trade-offs:** Model must match the architecture in `AlphaPanNet` exactly (no version mismatch handling needed for a single-author project). `torch.no_grad()` is already applied in `MCTS.search()` so no extra guard needed.

**Example:**
```python
# gui.py
import torch
from alpha_pan import Chenapan, AlphaPanNet, MCTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = Chenapan()
model = AlphaPanNet(device)
model.load_state_dict(torch.load("model_2.pt", map_location=device))
model.eval()

args = { 'C': 2, 'num_searches': 60, ... }  # same as training
mcts = MCTS(game, args, model)
```

### Pattern 3: Pygame Event Loop with AI Turn

**What:** In the pygame loop, distinguish human turns from AI turns. On human turn: wait for mouse click, validate the move, apply it. On AI turn: call `mcts.search(neutral_state)`, pick `argmax` of action probs, apply the move without blocking the event loop (AI moves are fast enough at 60 searches on CPU to not require threading for this scale).

**When to use:** Single-player human vs AI board game, where turns alternate and AI latency is acceptable inline.

**Trade-offs:** At `num_searches=60` on CPU, a single MCTS call takes 1-5 seconds per move. This freezes the pygame window during AI thinking. For this project this is acceptable (no animation needed during AI thinking). If smoother UX were needed, run MCTS in a `threading.Thread` and post a `pygame.USEREVENT` when done — but this adds complexity not required here.

**Example:**
```python
# Minimal play loop in gui.py
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if player == 1 and event.type == pygame.MOUSEBUTTONDOWN:
            # translate click → (start_pos, end_pos) action
            # validate against game.get_valid_moves()
            # apply via game.get_next_state()
            pass

    if player == -1:  # AI turn — blocking call acceptable
        neutral_state = game.change_perspective(state)
        action_probs = mcts.search(neutral_state)
        action = np.unravel_index(np.argmax(action_probs), action_probs.shape)
        neutral_state = game.get_next_state(neutral_state, action)
        state = game.change_perspective(neutral_state)
        player = game.get_opponent(player)

    draw_board(screen, state)
    pygame.display.flip()
```

### Pattern 4: Draw Penalty via `get_value_and_terminated`

**What:** The draw value is returned as `0` in the current code (line 376). Changing it to `-1` makes draws as bad as losses from the current player's perspective. This is a one-line change inside `Chenapan.get_value_and_terminated()`.

**When to use:** Whenever the training objective should discourage draws. The change propagates automatically through `selfPlay()` memory labeling — no other code needs to change.

**Trade-offs:** Draws now never provide a "safe" option. MCTS will value all non-win outcomes equally negatively, which is the intended behavior.

### Pattern 5: Configuration as Inline Dict (keep as-is)

**What:** The `args` dictionary at the bottom of `alpha_pan.py` holds all hyperparameters. It is passed by reference into `AlphaPan`, `MCTS`, and `Node`.

**When to use:** Single-file / two-file project with no deployment or config management requirements.

**Trade-offs:** No external config file needed. The same `args` values used for training should be used in `gui.py` for MCTS parameters (C, num_searches, dirichlet settings). Duplicate the dict in `gui.py` explicitly — do not rely on importing the module-level `args` from `alpha_pan` since that block is under `__main__` guard.

## Data Flow

### Training Flow

```
alpha_pan.py (main)
    │
    ├─ AlphaPan.learn()
    │     ├─ for each iteration:
    │     │     ├─ model.eval()
    │     │     ├─ selfPlay() × N games
    │     │     │     ├─ game.reset()
    │     │     │     ├─ loop: MCTS.search(state) → action_probs
    │     │     │     │         sample action → get_next_state()
    │     │     │     │         get_value_and_terminated()
    │     │     │     └─ terminal: label memory with outcome
    │     │     │
    │     │     ├─ model.train()
    │     │     ├─ train(memory) × num_epochs
    │     │     │     ├─ shuffle + batch
    │     │     │     ├─ forward: AlphaPanNet(state) → (policy, value)
    │     │     │     ├─ losses: BCE(policy) + MSE(value)
    │     │     │     └─ optimizer.step()
    │     │     └─ torch.save(model, optim)
    └─────┘
```

### Play Flow (GUI)

```
gui.py
    │
    ├─ load AlphaPanNet from model_N.pt
    ├─ model.eval()
    ├─ game.reset()
    ├─ pygame event loop:
    │     ├─ player == 1 (human):
    │     │     ├─ wait for MOUSEBUTTONDOWN
    │     │     ├─ resolve click → action (start, end)
    │     │     ├─ validate: action_end in valid_moves[action_start]
    │     │     └─ game.get_next_state(state, action)
    │     │
    │     ├─ player == -1 (AI):
    │     │     ├─ neutral_state = game.change_perspective(state)
    │     │     ├─ action_probs = mcts.search(neutral_state)  [blocking]
    │     │     ├─ action = argmax(action_probs)
    │     │     └─ state = change_perspective(get_next_state(neutral_state, action))
    │     │
    │     ├─ check get_value_and_terminated() → display result
    │     └─ draw_board(screen, state)
    └─────┘
```

### State Perspective Management (critical invariant)

The game always stores state from the "global" frame. Before MCTS for player -1, the state is flipped via `change_perspective()` so the neural net always sees the board from the current player's point of view. After the AI selects an action on `neutral_state`, the result is flipped back. This invariant must be replicated exactly in `gui.py` — the commented-out console play loop (lines 905–916) already shows the correct pattern.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Current (1 user, local) | Single file + gui.py. Blocking AI call acceptable at 60 searches. |
| Extended training (more iterations) | Add `tqdm` progress bars, structured logging to file, training config externalization. No structural change needed. |
| Faster play (more searches for stronger AI) | Run `mcts.search()` in a background thread; post `pygame.USEREVENT` when done. |
| Stronger AI (tournament quality) | Add model selection/evaluation loop (compare new vs old model). Structural change to training orchestrator, not GUI. |

## Anti-Patterns

### Anti-Pattern 1: Bare Module-Level Training Code

**What people do:** Leave `alphaPan.learn()` as a bare top-level call (as it currently exists in `alpha_pan.py`).

**Why it's wrong:** Importing `alpha_pan` in `gui.py` will immediately start training. The application becomes impossible to split.

**Do this instead:** Wrap the bottom block in `if __name__ == "__main__":`. This is the single most important structural change for the milestone.

### Anti-Pattern 2: Re-implementing Game Logic in the GUI

**What people do:** Duplicate move validation or state representation in the pygame layer to make click handling easier.

**Why it's wrong:** Two sources of truth for rules diverge. The existing `game.get_valid_moves()` and `game.get_next_state()` are already correct and tested implicitly through training.

**Do this instead:** Call `game.get_valid_moves(state, player)` on every turn in the GUI. Highlight valid destinations for the selected piece. Never duplicate the capture rules.

### Anti-Pattern 3: Training and GUI in the Same Process Branch

**What people do:** Add a flag like `PLAY_MODE = True` at the top of `alpha_pan.py` and branch inside the same file.

**Why it's wrong:** Mixes two distinct execution contexts with different event loops (pytorch training loop vs pygame event loop) and different lifecycle requirements. Hard to extend, hard to test.

**Do this instead:** Use `gui.py` as a separate entry point that imports core components from `alpha_pan`.

### Anti-Pattern 4: Applying Temperature Incorrectly During Self-Play

**What currently exists:** `temperature_action_probs` is computed but `action_probs` (without temperature) is used for sampling (lines 753–755 and 767–769 in `alpha_pan.py`). The temperature variable is dead code.

**Why it matters:** Temperature controls exploration during early self-play moves. At `temperature=1.25` the distribution should be flatter than raw visit counts, encouraging more diverse games early in training.

**Do this instead:** Sample from `temperature_action_probs / temperature_action_probs.sum()` instead of `action_probs`. Fix this in the same milestone as the draw penalty since both touch self-play behavior.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `gui.py` ↔ `alpha_pan.py` | Direct Python import | `gui.py` imports `Chenapan`, `AlphaPanNet`, `MCTS` |
| `MCTS` ↔ `AlphaPanNet` | Direct method call inside `torch.no_grad()` | Already decorated; safe for inference |
| `MCTS` ↔ `Chenapan` | Direct method calls | `get_valid_moves`, `get_next_state`, `get_encoded_state`, `get_value_and_terminated` |
| `gui.py` ↔ model checkpoints | `torch.load("model_N.pt")` | Hard-code latest checkpoint path or auto-detect highest N |
| `Chenapan` state ↔ pygame rendering | NumPy array read-only in renderer | `draw_board(screen, state)` reads `state[row, col]` directly — no conversion needed |

### GUI Click-to-Move Translation

The human click flow requires translating a pixel coordinate to a board position and then accumulating two clicks (start, end) to form an action. Suggested two-click state machine:

```
selected_start = None

on MOUSEBUTTONDOWN:
    pos = pixel_to_board(event.pos)   # → (row, col) → flat index 0-24
    if selected_start is None:
        if state[row, col] * player > 0:  # clicked own piece
            selected_start = pos
    else:
        action = [selected_start, pos]
        valid = game.get_valid_moves(state, player)
        if pos in valid[selected_start]:
            apply action
            selected_start = None
        else:
            selected_start = None  # deselect on invalid end
```

## Build Order Implications

1. **First:** Add `if __name__ == "__main__":` guard to `alpha_pan.py` — unlocks importability. Zero risk, one line of structural change.

2. **Second:** Change `return 0, True` to `return -1, True` in `Chenapan.get_value_and_terminated()` for the draw cases — the draw penalty. Test by verifying self-play still terminates.

3. **Third:** Fix the temperature sampling bug in `selfPlay()` — dead code removal + correct sampling. Same method, immediate neighbor to the draw penalty change.

4. **Fourth:** Build `gui.py` — pygame window, `draw_board`, click handler, AI turn dispatch. Depends on steps 1 and 2 being complete (importability + correct game outcomes).

5. **Optionally last:** Add structured training logging (replace `print` with `logging`, add loss tracking) — does not block GUI work, can be done in parallel with step 4 or after.

## Sources

- Direct source analysis of `alpha_pan.py` (932 lines, 2025 — HIGH confidence)
- [AlphaZero_Gomoku by junxiaosong](https://github.com/junxiaosong/AlphaZero_Gomoku) — reference multi-file AlphaZero architecture (`game.py`, `mcts_alphaZero.py`, `human_play.py`, `train.py` pattern — MEDIUM confidence)
- [alphazero-checkers-pygame by mlsdpk](https://github.com/mlsdpk/alphazero-checkers-pygame) — AlphaZero + pygame integration with strategy pattern for player types — MEDIUM confidence
- [pygame wiki: tut_design](https://www.pygame.org/wiki/tut_design) — MVC/Mediator pattern for pygame board games — MEDIUM confidence
- Python `__main__` guard: [official Python docs](https://docs.python.org/3/library/__main__.html) — HIGH confidence

---
*Architecture research for: AlphaZero self-play RL + pygame GUI (Alpha-Pan / Chénapan)*
*Researched: 2026-03-12*

# Project Research Summary

**Project:** Alpha-Pan
**Domain:** AlphaZero self-play reinforcement learning + pygame board game GUI (Python)
**Researched:** 2026-03-12
**Confidence:** HIGH (core bugs identified from direct codebase audit; architecture based on direct source analysis; stack locked by project constraints)

## Executive Summary

Alpha-Pan is an AlphaZero-style self-play RL system for the board game Chénapan (5x5), built in Python with PyTorch and NumPy. The codebase (~930 lines in `alpha_pan.py`) already contains the full RL pipeline — game engine (`Chenapan`), neural network (`AlphaPanNet`), MCTS with UCB selection, and a training loop — but has four confirmed bugs that make the current training code produce corrupted or meaningless results. These are not performance issues or design disagreements; they are correctness defects that must be fixed before any training run is worth keeping. The recommended approach is to fix all four bugs in a single Phase 1 pass, then build the pygame GUI as a separate `gui.py` file in Phase 2.

The stack is fully locked by project constraints (Python 3.10+, PyTorch 2.10, NumPy 2.x). The only open choices are supporting libraries: use `pygame-ce 2.5.7` (not upstream `pygame`) for the GUI, `tqdm` for training progress, and `torch.utils.tensorboard` for loss visualization. Hyperparameters in the existing code are set to toy values (`num_iterations=3`, `num_selfPlay_iterations=1`) that will never produce a trained model; these must be scaled to 50–200 iterations with 50–200 self-play games per iteration before training can be evaluated.

The highest-priority risk is discovering the four training bugs only after a long training run — recovery cost is HIGH because corrupted checkpoints cannot be salvaged and training must restart from scratch. All four bugs are one- to three-line fixes. They should be landed, verified, and committed before any training run begins. The second-priority risk is the pygame GUI blocking the OS event loop during AI computation (window freeze / "(Not Responding)"), which requires running MCTS in a background thread — this is an architectural decision that must be made during initial GUI design, not retrofitted later.

## Key Findings

### Recommended Stack

The project stack is fixed. The only actionable decisions are version selection and supporting utilities. Use `pygame-ce 2.5.7` (Community Edition) instead of upstream `pygame` — they share the same API but CE has active development and a more recent release cadence; they cannot coexist in the same environment. For training visibility, `tqdm` (already a dependency) handles progress bars and `torch.utils.tensorboard` (bundled with PyTorch) handles loss curves — no additional accounts or services needed.

One critical existing dependency issue: the current code uses `tqdm.notebook` which breaks in terminal scripts. This must be changed to plain `from tqdm import tqdm` for any standalone training run. For GPU users, `torch.compile(model)` after checkpoint load gives a 1.5–2x MCTS inference speedup at play time.

**Core technologies:**
- Python 3.10+: runtime — required by PyTorch 2.10
- PyTorch 2.10.0: neural network, training, inference — latest stable; `torch.compile` available
- NumPy 2.x: board state arrays, MCTS masks, policy arrays — already in use; compatible with torch 2.10
- pygame-ce 2.5.7: board rendering, mouse event handling — CE is strictly preferred over upstream pygame

**Supporting utilities:**
- tqdm 4.67.1: training loop progress bars — replace `tqdm.notebook` with plain `tqdm`
- tensorboard (2.x): loss visualization — use `torch.utils.tensorboard.SummaryWriter`, no extra install beyond `pip install tensorboard`

### Expected Features

The training system has four table-stakes bugs that prevent correct behavior, plus a configuration problem (toy hyperparameters). The GUI requires the full two-click board interaction pattern standard for pygame board games. None of the required features are complex — the bug fixes are one- to three-line changes, and the GUI components (board render, click handler, AI dispatch, game-over overlay) are individually straightforward.

**Must have — Training (P1):**
- Draw value = -1 (not 0) — collapses three-outcome MSE to binary, eliminates value head collapse
- Batch off-by-one fix (`len(memory)` not `len(memory)-1`) — last element of replay buffer was silently excluded
- Temperature applied to action sampling — exploration was dead code; wired but never used
- Deterministic board hashing — Python's `hash()` is session-randomized; breaks draw-by-repetition detection
- Per-iteration loss and game outcome logging — training is a black box without it
- Hyperparameter config block at top of file — current values are toy settings that produce no learning

**Must have — GUI (P1):**
- Board render (5x5 grid) + piece display
- Click-to-move input (two-click: select piece, then destination)
- Valid move highlighting after piece selection
- Turn indicator (Human / AI Thinking)
- AI move execution via MCTS with loaded checkpoint
- Game-over detection and display
- Restart without program relaunch

**Should have (P2, add after P1 is stable):**
- CSV loss log file — enables post-hoc trend analysis
- Last-move highlight in GUI — polish after basic play is functional
- "AI is thinking" indicator — add if AI think time is perceptibly long
- Draw rate trending (CSV) — meaningful only after enough training iterations

**Defer to v2+:**
- Zobrist hashing — only if profiling confirms hash() is the bottleneck
- Temperature schedule (exploration annealing) — add if training shows premature convergence
- Configurable search count via GUI

### Architecture Approach

The correct structure is two files: `alpha_pan.py` (unchanged core domain) and `gui.py` (new pygame entry point that imports from `alpha_pan`). The single most important structural change to `alpha_pan.py` is adding `if __name__ == "__main__":` around the training entry point at the bottom of the file — without this, importing `alpha_pan` in `gui.py` immediately starts training. The components are tightly coupled (MCTS takes `game` + `model`; trainer takes all four) at a scale (~930 lines) where splitting into many files adds import complexity without benefit.

**Major components:**
1. `Chenapan` (alpha_pan.py) — game rules, move validation, state transitions, draw conditions
2. `AlphaPanNet` (alpha_pan.py) — CNN dual-head: policy (25x25 probs) + value (scalar -1..1)
3. `MCTS` + `Node` (alpha_pan.py) — UCB selection, tree search, backpropagation
4. `AlphaPan` (alpha_pan.py) — training loop: self-play → replay buffer → batch SGD → checkpoint
5. `gui.py` (new) — pygame window, board rendering, click-to-move, AI turn dispatch via background thread

**Critical invariants:**
- State perspective must be flipped via `change_perspective()` before MCTS for player -1 and flipped back after. The correct pattern already exists in the commented-out console play loop (lines 905–916).
- `model.eval()` must be called immediately after every `torch.load()` in GUI code — never load a checkpoint without setting eval mode.
- The `args` dict used for play in `gui.py` must explicitly duplicate the training args — do not import module-level `args` from `alpha_pan` since that block is under the `__main__` guard.

### Critical Pitfalls

1. **Temperature not applied to action sampling** — `temperature_action_probs` is computed but raw `action_probs` is sampled instead. Fix: `np.random.choice(..., p=temperature_action_probs)`. Discovery after training = HIGH recovery cost (discard checkpoints, retrain).

2. **Draw value = 0 causes value head collapse** — with wins=+1, draws=0, losses=-1, the value head minimizes MSE by always predicting 0. Fix: return -1 for all draw branches in `get_value_and_terminated()`. Same HIGH recovery cost if discovered late.

3. **Non-deterministic board hashing breaks draw detection** — Python's `hash()` is session-randomized. Fix: replace with `hashlib.md5(state.tobytes())` or Zobrist hashing.

4. **Pygame event loop blocked by AI computation** — synchronous MCTS on the main thread freezes the window. Fix: run MCTS in a `threading.Thread`; post a custom `pygame.USEREVENT` on completion. Must be designed in from the start, not retrofitted.

5. **Missing `model.eval()` after checkpoint load** — dropout and batch norm behave incorrectly in training mode during inference. Fix: always call `model.eval()` immediately after `load_state_dict()`. Wrap in a `load_model_for_inference()` helper.

## Implications for Roadmap

Based on combined research, the project naturally splits into two phases with a clear dependency: the GUI cannot be meaningfully tested until at least one clean training run has produced a checkpoint, and a clean training run requires all four bugs to be fixed first.

### Phase 1: Training Bug Fixes and Pipeline Configuration

**Rationale:** All four training bugs (temperature, draw value, hash, batch slicing) corrupt the training signal. They are independent fixes but must all land before any training run is worth keeping. Recovery cost if discovered after training is HIGH. The hyperparameter scale-up and logging are not bugs but are required for training output to be meaningful. These belong in the same phase because they all gate the first valid training run.

**Delivers:** A correct, observable training pipeline capable of producing a checkpoint that reflects genuine learning.

**Addresses:**
- Draw value = -1 (one-line fix in `get_value_and_terminated()`)
- Temperature applied to sampling (one-line fix in `selfPlay()`)
- Deterministic board hashing (`hashlib.md5(state.tobytes())`)
- Batch off-by-one (`len(memory)` not `len(memory)-1`)
- Per-iteration loss and game outcome logging (tqdm + print/CSV)
- Hyperparameter config block (scale `num_iterations` to 50+, `num_selfPlay_iterations` to 50+)
- `if __name__ == "__main__":` guard (required for Phase 2 import)

**Avoids:** Value head collapse, corrupted replay buffer, non-reproducible draw detection, dead exploration code.

**Research flag:** No additional research needed. All fixes are confirmed from direct codebase audit with single-line resolutions.

### Phase 2: Pygame GUI

**Rationale:** Depends on Phase 1 for two things: (1) the `__main__` guard that allows `gui.py` to import `alpha_pan` without triggering training, and (2) a valid checkpoint to load. The GUI architecture must be decided upfront — specifically the threading model for AI computation — because retrofitting threading into a synchronous event loop is painful.

**Delivers:** A playable human-vs-AI interface for Chénapan with correct board rendering, click-to-move, valid move highlighting, and AI opponent via loaded checkpoint.

**Uses:** pygame-ce 2.5.7, MCTS with `model.eval()` inference, background threading for AI computation.

**Implements:** The `gui.py` architecture component with perspective-correct state management.

**Addresses:**
- Board render + piece display
- Two-click move input with valid move highlighting
- AI turn dispatch via background thread (avoids window freeze)
- `model.eval()` on checkpoint load (avoids silent inference corruption)
- QUIT handling every frame in all game states
- Auto-detect highest-numbered checkpoint (avoids loading stale model)
- Separate `play_args` from `train_args` (allows higher `num_searches` for play quality)

**Avoids:** Window freeze / "(Not Responding)", inference corruption from training-mode model, stale checkpoint loading.

**Research flag:** No additional research needed. All pygame patterns are well-documented and reference implementations (pygame chess projects, alphazero-checkers-pygame) provide verified patterns.

### Phase 3: Training Validation and Polish

**Rationale:** After Phase 1 fixes and Phase 2 GUI are in place, the next step is running actual training and confirming the model learns. This phase adds the observability tools that let you verify training is working and produces the P2 features.

**Delivers:** Confirmed-working training run with observable loss trends; GUI polish features; draw rate validation.

**Addresses:**
- CSV loss log file
- Draw rate trending (confirms draw=-1 is achieving aggressive play)
- Last-move highlight in GUI
- "AI is thinking" indicator
- Perspective flip verification test (forced-win trace)

**Avoids:** Silent training failures, difficulty distinguishing converging from broken training.

**Research flag:** No additional research needed for this phase. The draw rate behavior is MEDIUM confidence (drawn from AlphaZero.jl docs, not empirically verified for Chénapan specifically) — validate by running training and observing draw rate over iterations.

### Phase Ordering Rationale

- Phase 1 before Phase 2: The `__main__` guard is the architectural prerequisite for the two-file split. The bug fixes are the correctness prerequisite for a checkpoint worth playing against. Building the GUI before training works would mean testing against a broken model.
- Bugs before hyperparameters: The four correctness bugs should land before scaling hyperparameters, because a large training run on a broken pipeline wastes compute and produces misleading checkpoints.
- GUI architecture (threading) decided at start of Phase 2: The threading model affects every part of `gui.py` design. Retrofitting it after a synchronous implementation is significantly more work than designing it in from the start.
- CSV logging deferred to Phase 3: Print-based logging is sufficient to confirm Phase 1 is working. CSV adds value for multi-run trend analysis, which only matters once training is confirmed correct.

### Research Flags

Phases with standard patterns (no additional research needed):
- **Phase 1:** All fixes are confirmed from direct codebase audit. No research needed — implementation is mechanical.
- **Phase 2:** pygame board game patterns are well-documented. Threading model for AI computation is a standard solution. No research needed.
- **Phase 3:** No novel patterns. Logging is tqdm + CSV, which is standard.

No phases require `/gsd:research-phase` during planning. All key unknowns were resolved during project research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core stack locked by constraints. Versions verified against PyPI. pygame-ce recommendation based on release cadence comparison. |
| Features | HIGH (bugs) / MEDIUM (design) | Bug identification is HIGH — drawn from direct codebase audit confirmed against literature. Draw=-1 behavior is MEDIUM — theoretically sound (value collapse is documented), empirically unverified for this specific game. |
| Architecture | HIGH | Based on direct source analysis of 932-line codebase + verified reference AlphaZero implementations. Two-file structure rationale is sound for this scale. |
| Pitfalls | HIGH (critical) / MEDIUM (moderate) | Critical pitfalls (bugs 1-4) confirmed from source + literature. Moderate pitfalls (hyperparameter values, num_searches adequacy) are MEDIUM — drawn from reference implementations for similar-scale games, need empirical tuning for Chénapan specifically. |

**Overall confidence:** HIGH

### Gaps to Address

- **Hyperparameter values for Chénapan specifically:** The recommended training scale (50–200 iterations, 50–200 self-play games/iter) is drawn from reference implementations on similar small games (6x6 Othello, Connect Four). The specific values for a 5x5 game with Chénapan rules require empirical tuning. Plan for iterative adjustment after the first clean training run.

- **Perspective flip correctness:** Flagged as MEDIUM risk in CONCERNS.md. The labeling code's interaction with `change_perspective()` and the `update_meta_parameters` flag needs a forced-win trace test to verify it is correct. This is a verification gap, not a known bug.

- **Draw rate behavioral outcome:** Whether draw=-1 actually produces the intended aggressive play style is a testable hypothesis, not a certainty. The value collapse prevention mechanism is well-documented; the game-theoretic effect on Chénapan strategy is not. Monitor draw rate per iteration after Phase 1.

- **MCTS performance ceiling at 60 searches:** With a 5x5 board and 25x25=625 possible actions (most invalid), 60 simulations provide shallow tree coverage. Play quality impact is not empirically characterized. Consider testing with 200–400 searches in the GUI to assess the quality difference.

## Sources

### Primary (HIGH confidence)
- Direct source analysis of `alpha_pan.py` (932 lines, 2026) — all bug identification, component boundaries, data flow
- `.planning/codebase/CONCERNS.md` — batch off-by-one, hash bug, perspective flip risk, missing eval() pattern
- `.planning/PROJECT.md` — draw value design decision, temperature bug acknowledgment
- [torch · PyPI](https://pypi.org/project/torch/) — PyTorch 2.10.0, Python >=3.10 requirement
- [pygame-ce · PyPI](https://pypi.org/project/pygame-ce/) — version 2.5.7, released 2026-03-02
- [tqdm · PyPI](https://pypi.org/project/tqdm/) — version 4.67.1
- [PyTorch official docs — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — model.eval() requirement
- [Python `__main__` docs](https://docs.python.org/3/library/__main__.html) — entry point guard pattern
- [PyGame event loop documentation](https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_event_and_application_loop.md) — QUIT handling, event loop architecture

### Secondary (MEDIUM confidence)
- [leela-chess issue #20](https://github.com/glinscott/leela-chess/issues/20) — value collapse with draw=0 confirmation
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — reference hyperparameter scale (80 iterations, 100 episodes/iter for 6x6 Othello)
- [AlphaZero.jl training parameters](https://jonathan-laurent.github.io/AlphaZero.jl/dev/reference/params/) — batch size, temperature scheduling, draw penalty semantics
- [alphazero-checkers-pygame by mlsdpk](https://github.com/mlsdpk/alphazero-checkers-pygame) — AlphaZero + pygame integration patterns
- [AlphaZero_Gomoku by junxiaosong](https://github.com/junxiaosong/AlphaZero_Gomoku) — multi-file AlphaZero architecture reference
- [State of torch.compile August 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) — 1.5–2x inference speedup
- [arxiv:2003.05988](https://arxiv.org/abs/2003.05988) — hyperparameter sensitivity for small AlphaZero games

---
*Research completed: 2026-03-12*
*Ready for roadmap: yes*

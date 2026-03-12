# Feature Research

**Domain:** AlphaZero self-play training system + board game pygame GUI
**Researched:** 2026-03-12
**Confidence:** MEDIUM (training loop features verified via multiple AlphaZero implementations; GUI features verified via pygame chess projects; domain-specific draw-penalty behavior from training data only — LOW confidence)

---

## Feature Landscape

This document covers three subsystems: the training pipeline, the pygame GUI, and model evaluation. Each section separates table stakes (required for the system to work) from differentiators (valuable but not blocking) and anti-features (deliberately out of scope for this simple project).

---

### Table Stakes — Training Pipeline

Features the training system must have to produce a working, inspectable AI.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Draw value = -1 (not 0) | Core design requirement: AI must prefer winning over drawing; drives aggressive play style | LOW | One-line fix in self-play return value assignment; current code returns 0 for draw |
| Fix batch off-by-one | Training skips the final element of every batch (`len(memory)-1` instead of `len(memory)`) — causes silent data loss | LOW | Critical bug; fix is a single character change on line 797 |
| Fix temperature not applied | `temperature_action_probs` is computed but raw `action_probs` is sampled instead — exploration is broken | LOW | Must apply temperature to probs before `np.random.choice` call |
| Per-iteration progress output | Training runs blind without progress output — impossible to detect divergence, stalls, or bugs | LOW | Use `tqdm` (already a dependency); log iteration number, policy loss, value loss per batch |
| Self-play game count output | Show how many self-play games completed per iteration; visible signal that training is running | LOW | Print or tqdm.set_postfix with game count |
| Model checkpoint save per iteration | Already partially present; must reliably save `model_N.pt` and `optim_N.pt` after each learn phase | LOW | Existing pattern; verify it actually executes after every iteration |
| Deterministic board state hashing | Python's `hash()` is session-randomized (PYTHONHASHSEED); repeat-position detection (`biggest_loop`) is unreliable across runs | MEDIUM | Replace with `hashlib` or numpy-based hashing; required for correct draw detection |
| Configurable hyperparameters | All args hardcoded in script; impossible to tune without editing source | LOW | Move the `args` dictionary to a top-level config block or JSON file; keep single-file constraint |

### Table Stakes — Pygame GUI

Features any board game GUI must have to be playable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Board render (5x5 grid) | Visual representation of game state; minimum requirement for a GUI | LOW | Draw grid with `pygame.draw.rect`; map numpy board values to piece symbols or colors |
| Piece display on board | Show current pieces at their positions | LOW | Each cell shows piece type and ownership (color-coded or labeled) |
| Click-to-move input | Human player selects source then destination cell via mouse click | LOW | Two-click pattern: first click selects piece, second click sends move |
| Valid move highlighting | Show which destination squares are legal after source is selected | MEDIUM | Call `get_valid_moves()` on current state; highlight returned positions |
| Turn indicator | Display whose turn it is (Human / AI Thinking...) | LOW | Simple text label; prevents confusion during AI think time |
| AI move execution | Load trained model, run MCTS search, apply returned move | MEDIUM | Requires inference mode: load `model_N.pt`, instantiate MCTS, call search without training update |
| Game-over detection + display | Show win/loss/draw result when game ends; block further input | LOW | Check `get_value_and_terminated()`; display result overlay or status text |
| Restart capability | Allow starting a new game without relaunching the program | LOW | Reset game state and memory on keypress or button click |

### Table Stakes — Model Evaluation

Minimum visibility into whether training is working.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Loss logging per epoch | Policy loss + value loss per training batch; only signal of learning progress | LOW | `tqdm.set_postfix` or `print` inside the training loop |
| Loss trend visible over iterations | Aggregated average loss per full iteration (not just per batch) | LOW | Accumulate batch losses, print mean at end of each learn phase |
| Game outcome summary per self-play round | How many games ended in win/draw/loss; draw rate is the critical metric given draw=-1 design | LOW | Accumulate terminal values from `selfPlay()` return; print summary |

---

### Differentiators

Features that improve the system but are not required for it to work.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| CSV loss log file | Persists training history across sessions; enables post-hoc analysis and plotting | LOW | Open a CSV on train start, append row per iteration with timestamp + losses |
| Temperature schedule (exploration annealing) | Early training benefits from high temperature (exploration); later training converges faster with low temperature | MEDIUM | Piecewise schedule: high temp first N moves of each game, then greedy. Already partially implemented but not applied |
| Last-move highlight in GUI | Show the square the AI just moved to/from; aids human understanding | LOW | Store last action, highlight in a distinct color on render |
| "AI is thinking" indicator | Freeze UI and show spinner or label during MCTS search; prevents user confusion | LOW | Set a flag before MCTS call, render "AI is thinking..." text, force a display flip |
| Piece selection feedback (click highlight) | Highlight the piece the human has selected (first click); clarifies two-click move flow | LOW | Store selected square in state, draw distinct highlight on that cell |
| Draw rate trending | Track draw rate per iteration; reveals whether draw=-1 is achieving its goal of reducing draws | LOW | Extend game outcome summary to emit CSV row with draw_rate |
| Zobrist hashing for state | O(1) incremental state hashing vs O(n) string conversion; reduces MCTS overhead significantly | MEDIUM | Implement zobrist hash table at init; update incrementally on each move |
| Configurable MCTS search count via GUI | Allow human player to set num_searches before game start; faster for casual play, slower for stronger AI | LOW | Add a simple integer input or slider using pygame_gui or manual text rendering |

---

### Anti-Features (Deliberately Not Building)

| Feature | Why Avoid | What to Do Instead |
|---------|-----------|-------------------|
| Online multiplayer | Requires networking, session management, latency handling — far exceeds project scope | Local human vs AI only |
| AI vs AI visualization during training | Training is headless by design; rendering every self-play game would slow training drastically | Print game outcome summary per iteration as text |
| Move suggestion / hints | Adds inference complexity, exposes internals, muddies the "play against AI" experience | Play without hints; human learns by losing |
| Elo rating system | Requires playing AI versions against each other in an arena — substantial infrastructure for a single-model project | Track loss trend + draw rate as proxy for improvement |
| Move history panel | Requires move notation for a custom game (Chénapan is not chess/Go); effort for minimal value | Status text showing current turn and last move is sufficient |
| Animation / piece movement tweening | Smooth animation requires frame-level update loop changes; adds complexity for cosmetic benefit | Instant board redraw on move application |
| Drag-and-drop piece movement | Two-click (select + destination) is simpler to implement and equally functional for a 5x5 board | Two-click move input |
| Neural network architecture changes | ConvTranspose2d policy head is unusual but currently working; changing it risks breaking training | Fix bugs in existing pipeline, do not rearchitect the model |
| Unit test suite | Valuable long-term but not part of this milestone; scope is draw fix + training config + GUI | Add unit tests in a future milestone |

---

## Feature Dependencies

```
[Draw value fix (-1)]
    └──enables──> [Game outcome summary per self-play round]
                      └──enables──> [Draw rate trending]

[Deterministic board hashing]
    └──enables──> [Correct draw detection (biggest_loop)]
                      └──enables──> [Reliable game termination]

[Fix batch off-by-one]
    └──enables──> [Correct training loss values]

[Fix temperature not applied]
    └──enables──> [Proper exploration during self-play]

[Per-iteration loss logging]
    └──enhances──> [CSV loss log file]

[AI move execution (inference mode)]
    └──requires──> [Model checkpoint save per iteration]
    └──requires──> [Board render + Piece display]

[Valid move highlighting]
    └──requires──> [Click-to-move input] (need selected piece before highlights make sense)

[Game-over detection + display]
    └──requires──> [AI move execution]
    └──requires──> [Click-to-move input]
```

### Dependency Notes

- **Draw value fix requires no other feature:** It is a standalone one-line change, but it unblocks meaningful draw-rate tracking.
- **AI inference requires checkpoint:** The GUI needs a saved model to load; training must have run at least one iteration and saved a checkpoint.
- **Deterministic hashing is independent but urgent:** The biggest_loop mechanism is silently broken without it. Fix it alongside the draw value fix since both affect game termination correctness.
- **Valid move highlighting requires click state:** The GUI needs to track which square is currently selected before it can highlight valid destinations.

---

## MVP Definition

### Launch With (v1) — This Milestone

Minimum set to have a working, honest training system and a playable GUI.

- [ ] Draw penalty fix (draw returns -1) — core design requirement, one-line change
- [ ] Batch off-by-one fix — training is silently lossy without this
- [ ] Temperature applied to action probabilities — exploration is broken without this
- [ ] Deterministic board hashing — draw detection is unreliable without this
- [ ] Per-iteration loss + game outcome logging — training is a black box without this
- [ ] Hyperparameter config block at top of file — enables tuning without hunting through code
- [ ] Pygame GUI: board render + piece display + click-to-move + valid move highlights + turn indicator + game over display + restart
- [ ] AI inference mode in GUI: load checkpoint, run MCTS, apply move

### Add After Validation (v1.x)

- [ ] CSV loss log file — add once training is confirmed working; enables trend analysis
- [ ] Last-move highlight in GUI — polish once basic play is functional
- [ ] "AI is thinking" indicator — add if AI think time is perceptibly long during play testing
- [ ] Draw rate trending (CSV) — add once training has run enough iterations to generate meaningful data

### Future Consideration (v2+)

- [ ] Zobrist hashing — add if profiling confirms hashing is the bottleneck
- [ ] Temperature schedule — add if training quality analysis shows premature convergence
- [ ] Configurable search count via GUI — add if the AI feel is wrong during play testing

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Draw value = -1 | HIGH | LOW | P1 |
| Batch off-by-one fix | HIGH | LOW | P1 |
| Temperature fix | HIGH | LOW | P1 |
| Deterministic hashing | HIGH | MEDIUM | P1 |
| Per-iteration loss logging | HIGH | LOW | P1 |
| Hyperparameter config block | MEDIUM | LOW | P1 |
| Pygame GUI (core playability) | HIGH | MEDIUM | P1 |
| AI inference mode in GUI | HIGH | MEDIUM | P1 |
| CSV loss log | MEDIUM | LOW | P2 |
| Last-move highlight | LOW | LOW | P2 |
| "AI thinking" indicator | MEDIUM | LOW | P2 |
| Draw rate trending | MEDIUM | LOW | P2 |
| Zobrist hashing | LOW | MEDIUM | P3 |
| Temperature schedule | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for this milestone
- P2: Should have, add when P1 is stable
- P3: Future milestone consideration

---

## Confidence Notes

| Claim | Confidence | Source |
|-------|------------|--------|
| tqdm for per-iteration progress display | HIGH | tqdm PyPI docs + AlphaZero implementations |
| Two-click move input as GUI standard | HIGH | Multiple pygame chess implementations |
| Valid move highlighting as table stakes | HIGH | Pygame chess community (Chess on Python, GeeksforGeeks) |
| Draw = -1 drives aggressive play | MEDIUM | AlphaZero.jl docs (ternary_rewards), AlphaZero paper reward structure |
| Zobrist hashing as performance improvement | MEDIUM | CONCERNS.md audit + standard game programming practice |
| Temperature scheduling for exploration | MEDIUM | AlphaZero.jl training params (verified), paper |
| Deterministic hashing is broken with PYTHONHASHSEED | HIGH | CONCERNS.md audit + Python docs |

---

## Sources

- [AlphaZero.jl Training Parameters](https://jonathan-laurent.github.io/AlphaZero.jl/stable/reference/params/) — MEDIUM confidence (Julia implementation, but parameter semantics are canonical)
- [michaelnny/alpha_zero GitHub](https://github.com/michaelnny/alpha_zero) — MEDIUM confidence (PyTorch reference implementation with logging patterns)
- [tqdm PyPI](https://pypi.org/project/tqdm/) — HIGH confidence (official package docs)
- [How to Make a Chess Game with Pygame in Python](https://thepythoncode.com/article/make-a-chess-game-using-pygame-in-python) — MEDIUM confidence (practical GUI pattern source)
- [Build Chess with pygame — GeeksforGeeks](https://www.geeksforgeeks.org/python/create-a-chess-game-in-python/) — MEDIUM confidence (pygame board game feature reference)
- [Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play](https://liacs.leidenuniv.nl/~plaata1/papers/CoG2019.pdf) — MEDIUM confidence (academic, confirms loss tracking importance)
- `.planning/codebase/CONCERNS.md` — HIGH confidence (direct codebase audit)
- `.planning/PROJECT.md` — HIGH confidence (authoritative project requirements)

---
*Feature research for: AlphaZero self-play training + Chénapan pygame GUI*
*Researched: 2026-03-12*

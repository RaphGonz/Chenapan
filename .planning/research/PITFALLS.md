# Pitfalls Research

**Domain:** AlphaZero self-play training + pygame board game GUI
**Researched:** 2026-03-12
**Confidence:** HIGH (critical pitfalls drawn from code review + verified against literature)

---

## Critical Pitfalls

### Pitfall 1: Temperature Not Applied to Action Sampling (Already Present in Codebase)

**What goes wrong:**
`selfPlay()` computes `temperature_action_probs` but samples from the raw `action_probs` instead. During early training, all MCTS visit counts are near-uniform so this makes little difference. As the model matures and visit counts concentrate, the raw policy (without temperature) becomes near-deterministic. The AI stops exploring alternative lines, collapses to narrow repetitive play, and the training distribution becomes too homogeneous to generalize.

**Why it happens:**
The temperature variable was computed but never wired into the sampling call. It is a one-line oversight that is invisible until training is several hundred games deep.

**How to avoid:**
Replace `np.random.choice(..., p=action_probs)` with `np.random.choice(..., p=temperature_action_probs)`. Use temperature > 1 (e.g. τ=1.0) for the first N moves of each game to encourage exploration, then τ→0 for the remainder to sharpen the training signal.

**Warning signs:**
- Self-play games start looking identical after a few iterations
- Policy loss decreases steeply but value loss plateaus or oscillates
- Win rate against a fixed random baseline stops improving after early iterations

**Phase to address:** Phase 1 — Training Bug Fixes. This is the first fix to land because every subsequent training run is corrupted without it.

---

### Pitfall 2: Draw Reward = 0 Causes Value Head Collapse (Design Defect Being Fixed)

**What goes wrong:**
With wins=+1, draws=0, losses=-1, the mean training target is approximately 0. The value head discovers it can minimize MSE loss by ignoring the input and always predicting 0. Once stuck in this local minimum (around 33% accuracy), the value output stops providing meaningful signal to MCTS and the entire search quality degrades. This is a documented failure mode in public AlphaZero implementations (confirmed: github.com/glinscott/leela-chess issue #20).

**Why it happens:**
MSE on a balanced three-outcome distribution has a degenerate solution at the mean. The network finds this trivially.

**How to avoid:**
The PROJECT.md decision is correct: set draw value = -1. This collapses the target space to {-1, +1}, eliminating the degenerate zero-mean solution. The value head is now forced to distinguish losing from winning states rather than hiding behind a zero baseline.

**Warning signs:**
- Value loss drops quickly to a plateau around 0.5–0.65 and does not improve further
- Model always predicts values near 0 regardless of board position
- Self-play games run consistently to the move limit (50 moves) — the AI has no preference for winning early

**Phase to address:** Phase 1 — Training Bug Fixes. Must be changed before any meaningful training run.

---

### Pitfall 3: Perspective Flip Inconsistency in Value Target Labeling

**What goes wrong:**
In a two-player zero-sum game, the value target must be assigned from the perspective of the player who was to move at each stored state. If the perspective is flipped for the board encoding but not for the corresponding value label (or vice versa), the network learns an inverted value function for one player. The result is not a crash — training appears to proceed, loss decreases, but the model learns to prefer losing positions.

**Why it happens:**
The `change_perspective()` call in `get_next_state()` is gated by the `update_meta_parameters` flag. The labeling code at lines 781–787 assigns values using the `player` variable, but that variable is modified inside the loop. A trace-through is required to confirm the two paths always agree. This is flagged as medium-risk in CONCERNS.md.

**How to avoid:**
Before any training run, write a test: play a forced win from player=1, verify the stored value target for the winning terminal state is +1 for player 1's positions and -1 for player -1's positions. Run the same test with player=-1 starting. Both should produce symmetric, correctly-signed labels.

**Warning signs:**
- Player 1 and player -1 win rates are severely asymmetric after many games despite symmetric starting positions
- Value loss oscillates rather than decreases monotonically
- The model wins consistently as one color but loses as the other

**Phase to address:** Phase 1 — Training Bug Fixes, as a verification step after the draw-value fix.

---

### Pitfall 4: Non-Deterministic State Hashing Breaks Loop Detection (Already Present)

**What goes wrong:**
`get_hash()` uses Python's built-in `hash()` on a string. Python sets `PYTHONHASHSEED` randomly at startup by default. This means `hash(str(board))` produces different values across interpreter sessions. The `biggest_loop` counter (used to detect position repetition and trigger draws) counts visits against these unstable keys. A state visited 3 times in one session may only appear once in another, silently disabling the draw-by-repetition rule.

**Why it happens:**
`hash()` is the natural first instinct for hashing in Python. The randomization behavior is documented but not well-known outside of security contexts.

**How to avoid:**
Replace with deterministic hashing: `int(hashlib.md5(state.tobytes()).hexdigest(), 16)` or Zobrist hashing for O(1) incremental updates. Alternatively, set `PYTHONHASHSEED=0` in the launch environment as a stopgap, but fix the root cause.

**Warning signs:**
- Games that should draw by repetition occasionally play on past move 50 instead
- Self-play game lengths are inconsistent across runs with identical starting conditions
- The `biggest_loop` value in the encoded state (channel 1) rarely reaches 3

**Phase to address:** Phase 1 — Training Bug Fixes, because incorrect draw detection produces wrong game outcomes which corrupt the training signal.

---

### Pitfall 5: Pygame Event Loop Blocked by AI Computation — Window Freezes

**What goes wrong:**
MCTS with `num_searches=60` calls the neural network 60+ times per move. On CPU this takes several seconds. If the AI move computation runs synchronously on the main thread, `pygame.event.get()` is never called during this time. The OS marks the window as not responding. On Windows, the title bar grays out and the window cannot be moved, resized, or closed. Users cannot tell if the program has crashed or is thinking.

**Why it happens:**
The simplest GUI implementation calls MCTS directly from the event handler or game loop body, blocking until the move is ready.

**How to avoid:**
Run AI computation in a background thread (`threading.Thread`). The main loop continues to call `pygame.event.get()` and render a "thinking..." indicator every frame. When the thread completes, it posts a custom event (`pygame.event.post(pygame.event.Event(AI_MOVE_READY, {"action": action}))`) and the main loop processes it on the next tick.

Note: Python's GIL means this thread shares execution with the main thread, so you will not get parallelism — but for I/O-bound responsiveness (keeping the event loop alive) threading is sufficient. The AI thread still runs; the main thread just checks in between.

**Warning signs:**
- Window title says "(Not Responding)" during AI thinking
- Close button requires force-kill after the AI move completes
- `pygame.event.get()` is called fewer than once every ~200ms

**Phase to address:** Phase 2 — Pygame GUI. The threading pattern must be part of the initial GUI design, not retrofitted later.

---

### Pitfall 6: Missing `model.eval()` on Checkpoint Load — Silent Inference Corruption

**What goes wrong:**
PyTorch's `Dropout` and `BatchNorm` layers behave differently in training vs. inference mode. When a checkpoint is loaded for play (GUI mode), if `model.eval()` is not called, the model runs in training mode: dropout randomly zeros activations, and batch norm uses batch statistics instead of running statistics. The value and policy outputs vary on every call for the same position, making MCTS non-deterministic in a harmful way (not exploratory randomness — random noise from dropout).

**Why it happens:**
The current codebase has no inference interface (noted in CONCERNS.md as missing). When adding the GUI, checkpoint loading will be written from scratch and `model.eval()` is commonly forgotten.

**How to avoid:**
The loading sequence must be:
```python
model.load_state_dict(torch.load("model_N.pt", map_location=device))
model.eval()
```
Wrap this in a `load_model_for_inference()` helper so it cannot be called without eval mode.

**Warning signs:**
- The AI makes wildly inconsistent moves for the same board position on repeated calls
- Value estimates for the same state differ by more than 0.1 between consecutive queries
- Policy distribution is non-zero on positions that should be zero-probability

**Phase to address:** Phase 2 — Pygame GUI, specifically in the checkpoint-loading code.

---

## Moderate Pitfalls

### Pitfall 7: Off-by-One in Batch Slicing Silently Drops Training Data

**What goes wrong:**
`memory[batchIndex:min(len(memory)-1, batchIndex + args["batch_size"])]` uses `len(memory)-1` as the upper bound, which permanently excludes the last element of the replay buffer from training. With `batch_size=64`, the last 64 positions of every self-play game accumulate without ever contributing to gradient updates.

**Why it happens:**
The `-1` is a common off-by-one defensive pattern inherited from C-style indexing. Python slice notation is already exclusive at the upper end; the `-1` is wrong.

**How to avoid:**
Change to `min(len(memory), batchIndex + args["batch_size"])`. Add a unit test that verifies every element in a known memory list appears in exactly one batch.

**Phase to address:** Phase 1 — Training Bug Fixes.

---

### Pitfall 8: Insufficient Self-Play Games Per Iteration

**What goes wrong:**
The current `args["num_selfPlay_iterations"] = 1` means each training iteration produces data from a single game. A 5×5 game lasting 50 moves yields ~50 training samples. With `batch_size` likely at 64, training either skips the batch entirely or runs on a nearly empty batch, producing noisy gradient updates. The model will not converge.

**Why it happens:**
The args are set to minimal values for debugging/testing and were never updated for real training.

**How to avoid:**
For a 5×5 game of ~50 moves, target at minimum 25–50 self-play games per iteration to fill 1–2 full batches of 64. Published toy implementations (Connect Four, Othello, Tic-Tac-Toe) use 50–200 games per iteration. The exact number depends on game length and batch size: `num_games * avg_game_length >= 4 * batch_size` is a reasonable lower bound.

**Warning signs:**
- Training loss is extremely noisy with high variance between iterations
- Policy loss oscillates without a downward trend after 10+ iterations
- Running time per iteration is under 30 seconds on CPU (too fast = too little data)

**Phase to address:** Phase 1 — Training Pipeline Configuration.

---

### Pitfall 9: No Training Progress Logging Hides Silent Failures

**What goes wrong:**
Without logging policy loss, value loss, game outcomes, and iteration timing, there is no way to distinguish a silently broken training run from a converging one. The draw-value fix, perspective flip, and temperature fix can all be wrong simultaneously while the training loop runs to completion without error.

**Why it happens:**
Adding logging feels like polish, so it gets deferred. But for reinforcement learning, the training signal is the only feedback loop — without logging, bugs are invisible.

**How to avoid:**
Log per-iteration: policy loss, value loss, number of games played, mean game length, win/draw/loss counts from self-play. Even `print()` statements are better than nothing. Use `tqdm` (already imported) for batch-level progress. Consider a simple CSV log that can be plotted after training.

**Warning signs:**
- You do not know if loss went up or down during the last training run
- You cannot tell if self-play games are ending by win/draw/timeout

**Phase to address:** Phase 1 — Training Pipeline Configuration.

---

### Pitfall 10: Pygame Coordinate-to-Board-Cell Mapping Errors

**What goes wrong:**
Converting pixel coordinates from `pygame.mouse.get_pos()` to board cell indices requires knowing the exact pixel origin and cell size of the rendered grid. A common mistake is using the window size instead of the board area size, or forgetting that the board may be offset from pixel (0,0) by margins. The result is that clicks near cell boundaries register in the wrong cell.

**Why it happens:**
The board rendering code and the click-handling code are written separately and make independent assumptions about grid geometry. When the margin or cell size changes in one place, the other breaks.

**How to avoid:**
Define `CELL_SIZE`, `BOARD_ORIGIN_X`, `BOARD_ORIGIN_Y` as single constants. The conversion must be:
```python
col = (mouse_x - BOARD_ORIGIN_X) // CELL_SIZE
row = (mouse_y - BOARD_ORIGIN_Y) // CELL_SIZE
```
Both the renderer and the click handler must read from these same constants. Write a visual debug mode that highlights the cell under the cursor.

**Phase to address:** Phase 2 — Pygame GUI.

---

### Pitfall 11: Pygame Window Not Closeable During AI Turn

**What goes wrong:**
Even with threading in place, if `pygame.QUIT` is not checked in the main loop every frame, the window's close button queues the event but nothing processes it. The window stays open until the next frame after the AI finishes thinking.

**Why it happens:**
QUIT handling is often added only to the "waiting for human input" state, not as a global check at the top of every loop iteration.

**How to avoid:**
Check `pygame.QUIT` unconditionally at the start of every frame regardless of game state (waiting for human, AI thinking, game over). Call `pygame.quit()` then `sys.exit()` immediately on QUIT. This is the correct structure:
```python
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
    # ... other events only if not quit
```

**Phase to address:** Phase 2 — Pygame GUI.

---

## Minor Pitfalls

### Pitfall 12: Model Checkpoint Path Hardcoded, Wrong Checkpoint Loaded for Play

**What goes wrong:**
If the checkpoint filename is hardcoded (e.g. `"model_3.pt"`) and training produces `model_10.pt` after more iterations, the GUI loads an old, weaker model. The user does not notice because no version is displayed.

**How to avoid:**
Load the checkpoint with the highest iteration number by scanning for `model_*.pt` files and selecting the max. Display the loaded checkpoint name in the window title.

**Phase to address:** Phase 2 — Pygame GUI.

---

### Pitfall 13: `num_searches` Set Too Low for Meaningful MCTS

**What goes wrong:**
With `num_searches=60`, MCTS explores a limited tree. For a 5×5 board with up to 25×25=625 possible actions (though most are invalid), 60 simulations provide very shallow coverage. The AI's play quality is directly proportional to search depth. If the args are left at minimal values for the GUI demo, the AI will appear weak even after many training iterations.

**How to avoid:**
For play mode (not training), increase `num_searches` to 200–400. The GUI has no hard time constraint, so a 5–10 second think time for a visibly stronger move is an acceptable trade-off.

**Phase to address:** Phase 2 — Pygame GUI, in the inference configuration separate from the training args.

---

### Pitfall 14: `PYTHONHASHSEED` Not Pinned — Non-Reproducible Runs

**What goes wrong:**
Even after fixing the hash function, if `np.random.seed()` and `torch.manual_seed()` are not set, different training runs with the same args produce different results, making it impossible to confirm whether a code change improved or degraded training.

**How to avoid:**
Add a `seed` key to the args dictionary. At startup: `np.random.seed(args["seed"])`, `random.seed(args["seed"])`, `torch.manual_seed(args["seed"])`. Keep the seed optional (None = non-deterministic) for production training runs.

**Phase to address:** Phase 1 — Training Pipeline Configuration.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Everything in one file (`alpha_pan.py`) | No import complexity | Hard to unit-test components in isolation; GUI code mixed with training | Acceptable for this project's scope |
| Hardcoded `PYTHONHASHSEED` fix via env var | Zero code change | Does not fix correctness, just masks the symptom | Never — fix the hash function |
| `num_selfPlay_iterations=1` during development | Fast iteration | Runs that appear to train but produce noise | Only during smoke-testing, never for real runs |
| No `model.eval()` wrapper function | Saves 3 lines | Every caller must remember; one omission breaks inference | Never — wrap it |
| Print-based logging | Zero dependencies | Cannot plot, analyze, or compare runs | Acceptable for MVP training phase |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `hash(str(state))` per MCTS node | MCTS takes 2-5x longer than expected; profiler shows `str()` hotspot | Replace with `state.tobytes()` + `hashlib` or Zobrist | At `num_searches >= 100` |
| `list.count(h)` in move loop | Game slows progressively as move count increases | Use a `Counter` dict instead | Around move 30+ in a single game |
| `get_valid_moves()` called multiple times per state | CPU spikes during MCTS with no model call | Cache result per node after first computation | At `num_searches >= 50` |
| AI move on main thread | Window freeze during think time | Background thread + custom event | Every time AI searches >= 20 nodes |
| Unbounded memory list in self-play | RAM usage grows linearly with `num_selfPlay_iterations` | Cap buffer or use fixed-size deque | At `num_selfPlay_iterations >= 50` |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyTorch checkpoint → pygame GUI | Load with `torch.load()` only; forget `model.eval()` | Always: `load_state_dict()` then `model.eval()` |
| MCTS → pygame event loop | Call MCTS synchronously in event handler | Run MCTS in thread; post custom event on completion |
| Board state → pygame rendering | Use player perspective (which changes) for display | Render from a fixed canonical perspective (player 1 = bottom) |
| NumPy board array → pygame surface | Recompute full surface every frame | Only redraw changed cells or use dirty-rect updates |

---

## "Looks Done But Isn't" Checklist

- [ ] **Temperature fix:** `temperature_action_probs` is computed — verify it is actually used in `np.random.choice()`, not `action_probs`
- [ ] **Draw value:** `return 0` changed to `return -1` in all draw branches — verify ALL three draw conditions (move limit, repetition, no valid moves) return -1
- [ ] **Perspective value labels:** Stored training targets produce +1 for the winner's moves and -1 for the loser's moves — verify with a forced win trace test
- [ ] **Batch slicing:** `len(memory)-1` changed to `len(memory)` — verify last element appears in a batch via unit test
- [ ] **model.eval():** Called after every `torch.load()` in GUI code — search for every `load_state_dict` call and confirm `eval()` follows
- [ ] **QUIT handling:** `pygame.QUIT` checked every frame in all game states — verify by closing window during AI think time
- [ ] **Thread cleanup:** Background AI thread is joined or daemonized on exit — verify no zombie threads on window close
- [ ] **Hash determinism:** State hash produces same value for same board array across Python restarts — verify with a known board fixture

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Temperature bug — discovered after N training runs | HIGH | Discard all checkpoints; fix and retrain from scratch |
| Draw value = 0 — discovered after N training runs | HIGH | Discard all checkpoints; fix and retrain from scratch |
| Perspective flip — discovered after N training runs | HIGH | Cannot salvage corrupted checkpoints; retrain |
| Batch off-by-one — discovered after N training runs | MEDIUM | Fix and continue training; data loss is ~1/batch_size per pass |
| model.eval() missing — discovered in GUI | LOW | Add one line; reload checkpoint |
| Blocking event loop — discovered in GUI | LOW | Wrap MCTS call in thread; 1–2 hour refactor |
| Non-deterministic hash — discovered during testing | MEDIUM | Replace hash function; all existing game records are invalidated |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Temperature not applied | Phase 1: Bug Fixes | Unit test: sample distribution from `selfPlay()` is non-degenerate for a trained model |
| Draw value = 0 | Phase 1: Bug Fixes | Grep for `return 0` in draw branches; trace game to confirmed draw outcome |
| Perspective flip in value labels | Phase 1: Bug Fixes | Forced-win integration test with both player starting colors |
| Non-deterministic hash | Phase 1: Bug Fixes | `assert get_hash(board) == get_hash(board.copy())` across two interpreter sessions |
| Batch off-by-one | Phase 1: Bug Fixes | Unit test: all memory elements appear in batches |
| Insufficient self-play games | Phase 1: Pipeline Config | Log confirms `len(memory) >= 2 * batch_size` per iteration |
| No training progress logging | Phase 1: Pipeline Config | Loss values printed per iteration; CSV saved |
| model.eval() missing | Phase 2: GUI | Automated check: same board position returns identical value on two consecutive calls |
| AI blocking event loop | Phase 2: GUI | Manual test: window remains draggable during AI think time |
| Coordinate mapping errors | Phase 2: GUI | Click each corner cell; verify correct cell highlights |
| QUIT not handled every frame | Phase 2: GUI | Close window during AI think; verify immediate exit |
| Wrong checkpoint loaded | Phase 2: GUI | Window title shows loaded checkpoint filename |
| num_searches too low for play | Phase 2: GUI | Separate `play_args` from `train_args` in configuration |

---

## Sources

- CONCERNS.md codebase audit (2026-03-12) — batch off-by-one, hash bug, perspective flip risk, missing eval() pattern
- PROJECT.md project context (2026-03-12) — draw value design decision, temperature bug acknowledgment
- [leela-chess issue #20 — value collapse with draw=0](https://github.com/glinscott/leela-chess/issues/20) — MEDIUM confidence (GitHub issue, multiple participants confirming)
- [AlphaZero.jl PR #90 — temperature-adjusted MCTS policy as training target](https://github.com/jonathan-laurent/AlphaZero.jl/pull/90) — MEDIUM confidence (library maintainer discussion)
- [PyTorch official docs — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — HIGH confidence (official)
- [PyGame event loop documentation — Rabbid76/PyGameExamplesAndAnswers](https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_event_and_application_loop.md) — HIGH confidence (canonical community reference)
- [Fixing Pygame Window Freezing: Essential Event Handling Techniques](https://sqlpey.com/python/fixing-pygame-window-freezing-event-handling/) — MEDIUM confidence (verified against pygame docs)
- [Simple AlphaZero — Sura Nair](https://suragnair.github.io/posts/alphazero.html) — MEDIUM confidence (widely-cited implementation walkthrough)
- Training Parameters and Connect Four tutorial — [AlphaZero.jl official docs](https://jonathan-laurent.github.io/AlphaZero.jl/stable/reference/params/) — HIGH confidence

---
*Pitfalls research for: AlphaZero self-play (Python/PyTorch) + pygame board game GUI*
*Researched: 2026-03-12*

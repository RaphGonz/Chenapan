# Stack Research

**Domain:** AlphaZero-style self-play RL + pygame board game GUI (Python)
**Researched:** 2026-03-12
**Confidence:** HIGH (core stack is locked by project constraints; versions verified against PyPI)

## Context

The project constraints are fixed: Python + NumPy + PyTorch, no framework changes. The stack question is therefore not "what to use" but "what versions, what utilities, and what supporting libraries." This file focuses on:

1. Correct current versions of the locked dependencies
2. Supporting utilities for training (logging, progress, checkpointing)
3. pygame specifics for the board GUI
4. What to avoid and why

---

## Recommended Stack

### Core Technologies (Locked by Project Constraints)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ | Runtime | PyTorch 2.10 requires >=3.10; existing code uses f-strings, compatible with 3.10–3.13 |
| PyTorch | 2.10.0 | Neural network (AlphaPanNet), training, inference | Latest stable (released 2026-01-21); `torch.compile` available for inference speedup; proven for AlphaZero-style conv nets |
| NumPy | 2.x (latest) | Board state arrays, MCTS masks, policy arrays | Already used; PyTorch 2.10 ships with NumPy 2.x compatibility |
| pygame-ce | 2.5.7 | Board rendering, mouse event handling, GUI loop | Community Edition — more active development than upstream pygame 2.6.1, faster releases, same API, released 2026-03-02 |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | 4.67.1 | Training loop progress bars | Wrap iteration and epoch loops; use `tqdm.trange` for indexed loops. **Switch from `tqdm.notebook` to plain `tqdm`** if running outside Jupyter |
| tensorboard | latest (2.x) | Training metrics visualization (loss, value accuracy, policy entropy) | Use `torch.utils.tensorboard.SummaryWriter` — built into PyTorch; launch with `tensorboard --logdir runs/` |
| torch (built-in) | same as PyTorch | `torch.utils.tensorboard`, `torch.save`, `torch.load` | Already in use for checkpoints; no extra install needed |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Python venv | Isolate dependencies | `python -m venv .venv` then `pip install -r requirements.txt` |
| pip | Package manager | Already in use; no conda needed for this project |
| tensorboard CLI | Visualize training curves live | `pip install tensorboard`; run `tensorboard --logdir runs/` while training |

---

## Installation

```bash
# Core (verify existing environment first)
pip install torch==2.10.0
pip install numpy

# GUI — use pygame-ce, NOT pygame (more active, same API)
pip install pygame-ce==2.5.7

# Training utilities
pip install tqdm==4.67.1
pip install tensorboard
```

> **Note on pygame vs pygame-ce:** They cannot both be installed at the same time — they conflict. If `pygame` is already installed, uninstall it first: `pip uninstall pygame && pip install pygame-ce`.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| pygame-ce 2.5.7 | pygame 2.6.1 | Only if existing code has pygame-specific import quirks not fixed in CE; for a new GUI file, CE is strictly better |
| tqdm (plain) | tqdm.notebook | Only inside Jupyter notebooks; `tqdm.notebook.trange` breaks in terminal — **the existing code uses notebook variant, this must be changed for standalone training** |
| torch.utils.tensorboard | wandb | If experiment tracking across runs at scale is needed; for a single-machine personal project, tensorboard has zero account setup friction |
| torch.utils.tensorboard | matplotlib inline plots | Only for quick one-off visualization; TensorBoard handles live updates during training |
| torch.compile (optional) | eager mode | torch.compile gives 1.5–2x speedup for repeated forward passes; worth adding at inference time for the GUI (where MCTS calls the net hundreds of times per move) |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tqdm.notebook` in terminal scripts | Renders incorrectly outside Jupyter; produces broken output in plain Python scripts | `from tqdm import tqdm` or `tqdm.trange` |
| PyTorch Lightning / Ignite | Adds abstraction overhead for a training loop that is already written and working; would require restructuring `train()` and `learn()` | Plain PyTorch training loop as-is |
| Ray / multiprocessing for self-play | Over-engineering for a 5×5 board with fast MCTS; adds significant complexity | Single-threaded self-play is fine; training is the bottleneck, not data generation |
| tkinter for GUI | Inferior rendering performance vs pygame; no hardware-accelerated drawing; wrong tool for game boards | pygame-ce |
| OpenAI Gym / Gymnasium wrappers | Unnecessary abstraction; the game engine is already written as a custom class | Use the existing game API directly |
| pygame (upstream) | Slower development cadence; bugs fixed in CE not backported; last release was September 2024 | pygame-ce |

---

## Stack Patterns by Variant

**If running on CPU only (no CUDA):**
- Training will be slow; increase `num_selfPlay_iterations` incrementally to gauge time per iteration before committing to a long run
- `torch.compile` with `mode="reduce-overhead"` can help on CPU for repeated inference calls in MCTS

**If running on CUDA GPU:**
- Add `model = torch.compile(model)` after loading checkpoint for ~1.5x MCTS inference speedup
- Keep batch size at 64–256; larger batches don't help on this model size

**For the pygame GUI event loop:**
- Use `pygame.event.get()` with explicit `pygame.QUIT` and `pygame.MOUSEBUTTONDOWN` handling
- Cap frame rate with `clock.tick(60)` to avoid burning CPU while waiting for human input
- Call the AI synchronously on human turn submission (MCTS is fast enough on 5×5 that async is not needed)

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| torch 2.10.0 | Python 3.10–3.14 | Requires Python >=3.10; does NOT support Python 3.9 or earlier |
| pygame-ce 2.5.7 | Python 3.8–3.13 | Broader Python support than PyTorch; no conflict |
| numpy 2.x | torch 2.10.0 | PyTorch 2.0+ supports NumPy 2.x; avoid NumPy 1.x with torch 2.10 if possible |
| tqdm 4.67.1 | Python >=3.7 | No version conflicts with the rest of the stack |
| tensorboard | Python >=3.8 | Works with any torch 2.x; install separately, not bundled |

---

## Key Training Hyperparameter Guidance (from research)

The existing hyperparameters in `alpha_pan.py` are very conservative for initial testing but need scaling for real training:

| Parameter | Current Value | Recommended for Real Training | Rationale |
|-----------|---------------|-------------------------------|-----------|
| `num_iterations` | 3 | 50–200 | 3 iterations produces no meaningful learning; AlphaZero-style training needs hundreds of self-play/train cycles |
| `num_selfPlay_iterations` | 1 | 50–200 games/iter | 1 game per iteration starves the replay buffer |
| `num_searches` (MCTS) | 60 | 60–200 | 60 is reasonable for 5×5; go higher for stronger play |
| `num_epochs` | 1 | 4–10 | More epochs per batch of data before replacing with new self-play data |
| `batch_size` | 64 | 128–512 | Larger batches stabilize gradient estimates |
| Learning rate | not explicit | 1e-3 with weight decay 1e-4 | Standard for AlphaZero; add to Adam/SGD optimizer config |
| Draw value | 0 (implicit) | **-1 (same as loss)** | Core PROJECT.md requirement: forces aggressive play, eliminates draw-seeking strategies |

**Confidence on hyperparameters:** MEDIUM — drawn from alpha-zero-general reference implementations and the AlphaZero.jl documentation; specific values for a 5×5 game require empirical tuning.

---

## Sources

- [torch · PyPI](https://pypi.org/project/torch/) — PyTorch 2.10.0 verified current stable, Python >=3.10 (HIGH confidence)
- [pygame-ce · PyPI](https://pypi.org/project/pygame-ce/) — version 2.5.7, released 2026-03-02 (HIGH confidence)
- [pygame · PyPI](https://pypi.org/project/pygame/) — version 2.6.1, released 2024-09-29 (HIGH confidence)
- [tqdm · PyPI](https://pypi.org/project/tqdm/) — version 4.67.1 confirmed (HIGH confidence)
- [torch.utils.tensorboard — PyTorch 2.10 docs](https://docs.pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API confirmed stable (HIGH confidence)
- [pygame-community/pygame-ce GitHub](https://github.com/pygame-community/pygame-ce) — CE vs upstream comparison (MEDIUM confidence — community source)
- [AlphaZero.jl training parameters](https://jonathan-laurent.github.io/AlphaZero.jl/dev/reference/params/) — batch size 2048, cpuct=1.0, temperature scheduling (MEDIUM confidence — Julia port, not Python, but based on original paper)
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — 25 MCTS sims for 6x6 Othello, 80 iterations, 100 episodes/iter (MEDIUM confidence)
- [arxiv:2003.05988](https://arxiv.org/abs/2003.05988) — hyperparameter sensitivity for small AlphaZero games; self-play iterations subsume MCTS/episode counts (MEDIUM confidence)
- [State of torch.compile August 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) — 1.5–2x speedup typical (MEDIUM confidence)

---
*Stack research for: Alpha-Pan (AlphaZero-style self-play RL + pygame GUI)*
*Researched: 2026-03-12*

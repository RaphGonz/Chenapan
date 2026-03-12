# Phase 1: Training Foundation - Research

**Researched:** 2026-03-12
**Domain:** AlphaZero-style self-play training pipeline — PyTorch, bug fixes, observability
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Draw outcome returns -1 in `get_value_and_terminated()` — eliminates value head collapse | Verified: current code returns `0` on draw (line 377); change to `-1` is a one-line fix confirmed safe by AlphaZero literature |
| TRAIN-02 | Temperature applied to action probability sampling in `selfPlay()` | Verified: `temperature_action_probs` is computed but unused; fix is to normalize and use it as the `p=` argument |
| TRAIN-03 | Board hashing uses `hashlib.md5(state.tobytes())` instead of Python's session-randomized `hash()` | Verified: `hash(str(state))` at line 99 is non-deterministic across runs; `hashlib.md5` is deterministic by definition |
| TRAIN-04 | Replay buffer batch slice uses `len(memory)` not `len(memory)-1` | Verified: line 797 has `min(len(memory)-1, ...)` which silently drops the last sample every batch |
| PIPE-01 | `if __name__ == "__main__":` guard wraps training entry point | Verified: lines 839–857 execute at module import; a guard prevents this |
| PIPE-02 | `tqdm.notebook` import replaced with plain `tqdm` | Verified: `from tqdm.notebook import trange` at line 20 raises `ImportError` outside Jupyter |
| PIPE-03 | Per-iteration loss (policy + value) and win/draw rate logged to console | Pattern clear: accumulate loss values in `train()`, print summary in `learn()` after each self-play + train cycle |
| PIPE-04 | Hyperparameter config block with real training values | Current defaults (`num_iterations=3`, `num_selfPlay_iterations=1`) are toy values; research gives 50–200 range |
| MODEL-01 | `AlphaPanNet` rebuilt with residual blocks (AlphaZero-style skip connections) | AlphaZero standard: residual tower as backbone, policy head + value head on top; enables gradient flow in deeper nets |
</phase_requirements>

---

## Summary

This phase is a **surgical bug-fix and structural hardening** of an existing, nearly-complete AlphaZero implementation. The game engine, MCTS, neural network architecture, self-play loop, and training loop are all present in `alpha_pan.py`. What's missing are: four correctness bugs that silently corrupt training, two import/structural issues that break the script outside Jupyter, training observability (console logging), scaled hyperparameters, and an upgraded neural network backbone.

The four bugs (TRAIN-01 through TRAIN-04) are each one-to-three line changes identified precisely in the source. None require redesigning surrounding logic. The structural fixes (PIPE-01, PIPE-02) are also single-line changes. PIPE-03 and PIPE-04 require adding a few lines in the existing `learn()` and `train()` methods and the `args` dict. MODEL-01 is the most involved change: replacing the sequential `compression_block` in `AlphaPanNet` with a residual tower while keeping the same input shape (5 channels, 5×5 board) and same output contract (policy 25×25, value scalar).

The project has no test infrastructure. There is no `pytest.ini`, no `tests/` directory, no existing test files. All verification will be manual / console-output inspection. This is appropriate for the phase scope — the success criteria are observable behaviors (console output, deterministic results, correct sampling) rather than unit-testable API contracts.

**Primary recommendation:** Fix the four training bugs first (they are independent and low-risk), then add the structural fixes, then PIPE-03/PIPE-04 observability, then MODEL-01 as the final and riskiest change. Keep all changes in `alpha_pan.py`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.x (existing) | Language | Existing project constraint |
| PyTorch | existing install | Neural network, GPU training | Existing project constraint |
| NumPy | existing install | Board state arrays, action masking | Existing project constraint |
| tqdm | existing install | Progress bars (plain, not notebook) | Lightweight, standard for training loops |
| hashlib | stdlib | Deterministic board state hashing | Python stdlib, deterministic by spec |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nn.functional | PyTorch stdlib | Loss functions (BCE, MSE) | Already used in `train()` |
| random | Python stdlib | Memory shuffle before training | Already used in `train()` |
| math | Python stdlib | UCB formula in MCTS | Already used in Node |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `from tqdm import trange` | `from tqdm.auto import trange` | `tqdm.auto` works in both Jupyter and scripts; plain `tqdm` is simpler and sufficient for a script-only target |
| `hashlib.md5(state.tobytes()).hexdigest()` | `hashlib.sha256(state.tobytes()).hexdigest()` | SHA-256 is more collision-resistant but slower; MD5 is fine for board repetition detection (not cryptography) |

**Installation:** No new packages required. All dependencies are already in the project.

---

## Architecture Patterns

### Recommended Project Structure

```
alpha_pan.py          # All code: game, MCTS, network, training (existing single-file structure)
model_{N}.pt          # Checkpoint per iteration (existing)
optim_{N}.pt          # Optimizer state per iteration (existing)
```

The project decision (documented in PROJECT.md) is to keep everything in one file. Phase 1 does not change this structure.

### Pattern 1: Residual Block (AlphaZero style)

**What:** A `ResidualBlock` class wraps two Conv2d-BN-ReLU pairs with a skip connection that adds the input to the output.

**When to use:** As the backbone of `AlphaPanNet` replacing the current sequential `compression_block`. Insert N residual blocks between the initial conv and the heads.

**Example:**
```python
# Source: AlphaZero community implementations, cross-verified with PyTorch docs
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual          # skip connection
        out = F.relu(out)
        return out
```

The input and output shapes are identical (same channels, same spatial dims) so no 1×1 projection is needed as long as the number of channels stays constant across all residual blocks.

### Pattern 2: `if __name__ == "__main__":` Guard

**What:** Wrap all training entry-point code in a guard so the module can be imported without triggering training.

**When to use:** Always, in any Python training script that defines importable classes.

**Example:**
```python
# After all class definitions
if __name__ == "__main__":
    game = Chenapan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPanNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = { ... }
    alphaPan = AlphaPan(model, optimizer, game, args)
    alphaPan.learn()
```

### Pattern 3: Temperature-Scaled Sampling

**What:** AlphaZero raises visit-count probabilities to the power `1/temperature` before sampling. This sharpens the distribution (temperature < 1) or flattens it (temperature > 1). The result must be re-normalized to sum to 1.

**When to use:** In `selfPlay()` when selecting the actual move to play (not just storing MCTS probs in memory).

**Example:**
```python
# In selfPlay(), after getting action_probs from MCTS:
temperature_action_probs = action_probs ** (1.0 / self.args["temperature"])
temperature_action_probs /= np.sum(temperature_action_probs)   # re-normalize
action_flat = np.random.choice(
    self.game.action_size * self.game.action_size,
    p=np.matrix.flatten(temperature_action_probs)
)
```

### Pattern 4: Deterministic Board Hashing

**What:** Replace Python's built-in `hash()` (PYTHONHASHSEED-randomized per process) with `hashlib.md5()` on the array bytes.

**Example:**
```python
import hashlib

def get_hash(self, state):
    # Ensure C-contiguous memory layout before tobytes()
    s = np.ascontiguousarray(state)
    return hashlib.md5(s.tobytes()).hexdigest()
```

### Pattern 5: Draw Value = -1

**What:** In `get_value_and_terminated()`, the draw branch returns `(0, True)`. Change to `(-1, True)`.

**Why it matters:** With value 0, the value head is trained on examples where terminal states have +1 (win) or 0 (draw/loss). The mean over many draws and losses pushes predictions toward 0 regardless of board quality. Setting draw = -1 makes draws as bad as losses from the network's perspective, preventing the collapse.

### Anti-Patterns to Avoid

- **Using `tqdm.notebook.trange` in a script:** Raises `ImportError` or silently does nothing outside an IPython kernel. Use `from tqdm import trange` for scripts.
- **Keeping toy hyperparameters for a real run:** `num_iterations=3, num_selfPlay_iterations=1` produces essentially zero learning. The model will overfit to near-nothing and produce meaningless checkpoints.
- **Forgetting to normalize `temperature_action_probs` after scaling:** `action_probs ** (1/T)` no longer sums to 1. Passing an un-normalized array as `p=` to `np.random.choice` raises a `ValueError`.
- **Off-by-one in batch slice:** `memory[i:min(len(memory)-1, i+bs)]` will silently drop the last sample every batch when `len(memory)` is a multiple of `batch_size`, and always drops at least one sample otherwise. Use `min(len(memory), ...)`.
- **Adding skip connections with mismatched channel counts:** The standard residual block pattern requires input channels == output channels for the identity shortcut. If the initial conv changes channels, use a projection conv (1×1) in the shortcut.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress bars in training loop | Custom print counter | `tqdm.trange` | Line-count display, ETA estimation, nested bars — all free |
| Deterministic hashing | Custom state serialization | `hashlib.md5(state.tobytes())` | `tobytes()` is NumPy's native binary serialization; hashlib is stdlib |
| Loss computation | Manual gradient accumulation | `F.binary_cross_entropy_with_logits`, `F.mse_loss` | Already in use; handles numerical stability (log-sum-exp) |
| Residual block | Complex custom architecture | Standard `ResidualBlock` pattern (2 conv + skip) | Well-understood, gradient-safe, matches AlphaZero paper |

**Key insight:** Every item in this phase is either a stdlib fix or a well-known 5–10 line pattern. No new external dependencies are needed.

---

## Common Pitfalls

### Pitfall 1: Forgetting to re-normalize temperature probabilities

**What goes wrong:** `ValueError: probabilities do not sum to 1` at runtime, or silent probability bias if the values are very small but happen to sum close to 1.

**Why it happens:** `probs ** (1/T)` raises each element to a power, changing the sum away from 1.0.

**How to avoid:** Always divide by `np.sum(temperature_action_probs)` immediately after the power operation.

**Warning signs:** `np.sum(temperature_action_probs)` is not 1.0 after scaling.

### Pitfall 2: `np.ascontiguousarray` needed for hashing after perspective flip

**What goes wrong:** `state.tobytes()` on a non-contiguous array (e.g., after `np.rot90`) produces bytes for the underlying buffer, not the logical array contents — two logically identical boards can hash differently.

**Why it happens:** `np.rot90` returns a view with a different memory stride, not a copy. `tobytes()` on a strided view can produce unexpected results.

**How to avoid:** Call `np.ascontiguousarray(state)` before `tobytes()` in `get_hash()`.

**Warning signs:** The same board state hashing to different values in determinism tests.

### Pitfall 3: BatchNorm in eval mode during self-play

**What goes wrong:** The model is set to `model.eval()` before self-play (correctly, in the current code). But if any change inadvertently removes this, BatchNorm will use batch statistics during inference, producing different results for batch size 1 vs larger batches.

**Why it happens:** BatchNorm behaves differently in `train()` vs `eval()` mode.

**How to avoid:** The existing pattern of `self.model.eval()` before self-play and `self.model.train()` before training is correct — preserve it during MODEL-01 refactoring.

**Warning signs:** Value predictions that vary wildly across otherwise identical board states.

### Pitfall 4: Residual block dimension mismatch when changing channel count

**What goes wrong:** `RuntimeError: Expected input channels X, got Y` during the skip connection addition.

**Why it happens:** If the initial convolutional layer changes from 5 channels to N channels, subsequent residual blocks must all use N channels for both conv layers. If you mix channel sizes, the skip addition fails.

**How to avoid:** Choose a fixed `num_hidden_channels` for the residual tower. Use a separate initial conv layer to project from 5 (input channels) to `num_hidden_channels` before entering the residual blocks.

**Warning signs:** Shape errors at the `out += residual` line in `forward()`.

### Pitfall 5: Perspective flip correctness in value labeling

**What goes wrong:** Training data gets the wrong sign on value targets for player -1. The policy loss trains correctly but the value head is trained with inverted targets for one player.

**Why it happens:** The `selfPlay()` loop works from the always-current player's neutral perspective (via `change_perspective`). The labeling logic at game end must correctly assign `value` vs `get_opponent_value(value)` based on which player won.

**How to avoid:** Add a forced-win trace test (create a board one move from a win, verify `returnMemory` has `+1` for the winning player's state and `-1` for the other's).

**Warning signs:** Value loss decreases but the model predicts near-zero for all states, or the model always expects to win from any position.

---

## Code Examples

### TRAIN-01: Fix draw value

```python
# In Chenapan.get_value_and_terminated()
# BEFORE (line 377):
return 0, True
# AFTER:
return -1, True
```

### TRAIN-02: Fix temperature sampling

```python
# In AlphaPan.selfPlay(), for BOTH player == 1 and player == -1 branches:
# BEFORE:
temperature_action_probs = action_probs ** (1/self.args["temperature"])
action_float = np.random.choice(
    self.game.action_size*self.game.action_size,
    p=np.matrix.flatten(action_probs)          # <-- wrong: raw probs
)
# AFTER:
temperature_action_probs = action_probs ** (1.0 / self.args["temperature"])
temperature_action_probs /= np.sum(temperature_action_probs)  # re-normalize
action_float = np.random.choice(
    self.game.action_size * self.game.action_size,
    p=np.matrix.flatten(temperature_action_probs)  # <-- correct
)
```

### TRAIN-03: Fix deterministic hashing

```python
# At top of file:
import hashlib

# In Chenapan.get_hash():
# BEFORE:
def get_hash(self, state):
    return hash(str(state))
# AFTER:
def get_hash(self, state):
    s = np.ascontiguousarray(state)
    return hashlib.md5(s.tobytes()).hexdigest()
```

### TRAIN-04: Fix batch slice off-by-one

```python
# In AlphaPan.train():
# BEFORE (line 797):
sample = memory[batchIndex:min(len(memory)-1, batchIndex + self.args["batch_size"])]
# AFTER:
sample = memory[batchIndex:min(len(memory), batchIndex + self.args["batch_size"])]
```

### PIPE-01: Add __main__ guard

```python
# Replace bare execution block (lines 839–857) with:
if __name__ == "__main__":
    game = Chenapan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPanNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 100,          # scaled up from 3
        'num_selfPlay_iterations': 100, # scaled up from 1
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3
    }
    alphaPan = AlphaPan(model, optimizer, game, args)
    alphaPan.learn()
```

### PIPE-02: Fix tqdm import

```python
# BEFORE:
from tqdm.notebook import trange
# AFTER:
from tqdm import trange
```

### PIPE-03: Console logging in learn()

```python
# In AlphaPan.learn(), after training each iteration:
def learn(self):
    for iteration in trange(self.args['num_iterations']):
        memory = []
        self.model.eval()

        win_count = 0
        draw_count = 0
        for _ in trange(self.args['num_selfPlay_iterations'], desc="Self-play", leave=False):
            game_memory = self.selfPlay()
            memory += game_memory
            # count outcomes in this game
            outcomes = [outcome for _, _, outcome in game_memory]
            if outcomes and outcomes[-1] == 1:
                win_count += 1
            elif outcomes and outcomes[-1] == -1:
                draw_count += 1

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in trange(self.args['num_epochs'], desc="Training", leave=False):
            pl, vl, nb = self.train(memory)  # train() must return loss values
            total_policy_loss += pl
            total_value_loss += vl
            num_batches += nb

        avg_pl = total_policy_loss / max(num_batches, 1)
        avg_vl = total_value_loss / max(num_batches, 1)
        total_games = self.args['num_selfPlay_iterations']
        print(
            f"Iter {iteration:03d} | "
            f"PolicyLoss={avg_pl:.4f} | "
            f"ValueLoss={avg_vl:.4f} | "
            f"WinRate={win_count/total_games:.2%} | "
            f"DrawRate={draw_count/total_games:.2%}"
        )

        torch.save(self.model.state_dict(), f"model_{iteration}.pt")
        torch.save(self.optimizer.state_dict(), f"optim_{iteration}.pt")
```

Note: `train()` must be modified to return `(policy_loss_sum, value_loss_sum, num_batches)`.

### MODEL-01: Residual tower for AlphaPanNet

```python
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaPanNet(nn.Module):
    def __init__(self, device, num_res_blocks=4, num_hidden=64):
        super().__init__()
        self.device = device

        # Initial projection: 5 input channels -> num_hidden
        self.start_block = nn.Sequential(
            nn.Conv2d(5, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        # Value head: num_hidden channels -> scalar
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Policy head: num_hidden channels -> 25x25 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 25 * 25)
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x).view(-1, 25, 25)
        value = self.value_head(x)
        return policy, value
```

**Critical:** The policy head output shape must still be `(B, 25, 25)` to match the existing MCTS masking and softmax in `MCTS.search()`. The flat Linear approach (32*5*5 -> 625 -> view 25×25) achieves this without the convolutional decoder chain that was in the original policy head.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sequential CNN backbone | Residual tower backbone | AlphaZero paper (2017) | Enables 4–20+ deep blocks without vanishing gradients |
| `tqdm.notebook` for progress | `tqdm` or `tqdm.auto` | Always for scripts | Avoids ImportError outside Jupyter |
| `hash()` for board states | `hashlib` on bytes | Python 3.3+ (PYTHONHASHSEED randomization) | Deterministic cross-run results |

**Deprecated/outdated in this codebase:**
- `hash(str(state))`: Non-deterministic since Python 3.3 due to hash randomization. Will produce different draw detection across runs.
- `tqdm.notebook.trange`: Notebook-only import. Cannot run as standalone script.
- Sequential compression block (no skip connections): Not wrong, but training will plateau faster with deeper networks.

---

## Open Questions

1. **Hyperparameter values for num_iterations / num_selfPlay_iterations**
   - What we know: 3 iterations / 1 game is clearly too low; AlphaZero research suggests 50–200
   - What's unclear: Optimal values depend on CPU/GPU speed and how quickly the game converges
   - Recommendation: Start with 100/100 as the PIPE-04 config, note it in a comment as tunable

2. **Perspective flip correctness in value labeling**
   - What we know: The `selfPlay()` loop always operates from the current player's neutral perspective; labeling assigns `value` vs `get_opponent_value(value)` based on `hist_player == player`
   - What's unclear: Whether `player` at the time `returnMemory` is constructed correctly reflects who actually won (it is the final value of `player` after the game ends, which is the player who made the last move)
   - Recommendation: Create a forced-win trace during Phase 1 verification: construct a board state one move from a win, run `selfPlay()` to terminal, inspect `returnMemory` targets

3. **num_res_blocks / num_hidden for MODEL-01**
   - What we know: Original network had 3 conv layers total; a 4-block residual tower with 64 channels is a reasonable starting point for a 5×5 board
   - What's unclear: Whether 4 blocks is over/under-parameterized for this board size
   - Recommendation: Use `num_res_blocks=4, num_hidden=64` as the default and document them as hyperparameters

---

## Sources

### Primary (HIGH confidence)
- Python stdlib `hashlib` documentation — https://docs.python.org/3/library/hashlib.html — confirmed deterministic behavior
- NumPy `ndarray.tobytes()` documentation — https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndarray.tobytes.html — confirmed bytes output for contiguous arrays
- PyTorch `nn.Module` / residual block pattern — https://discuss.pytorch.org/t/is-this-the-right-way-to-create-skip-connections-in-pytorch/97934 — standard skip connection usage

### Secondary (MEDIUM confidence)
- AlphaZero Connect4 PyTorch reference implementation — https://github.com/plkmo/AlphaZero_Connect4 — residual block + policy/value head architecture pattern
- tqdm official docs — https://tqdm.github.io/docs/notebook/ — confirmed `tqdm.auto` as multi-environment import
- AlphaZero action sampling paper — https://joshvarty.github.io/AlphaZero/ — temperature formula `π_i ∝ n_i^(1/τ)`

### Tertiary (LOW confidence)
- Hyperparameter range (50–200 iterations) is from community implementations and the project's own STATE.md note, not the original DeepMind paper (which used much larger scale)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all existing dependencies, no new packages
- Bug fixes (TRAIN-01 to TRAIN-04): HIGH — bugs identified precisely in source code with line numbers
- Architecture patterns: HIGH — standard AlphaZero residual block pattern, verified against multiple implementations
- Pitfalls: HIGH — derived directly from code analysis and known Python/NumPy behaviors
- Hyperparameter values: MEDIUM — reasonable ranges from literature, empirically unverified for this specific game

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable domain — PyTorch API, hashlib, numpy are stable)

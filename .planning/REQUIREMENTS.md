# Requirements: Alpha-Pan

**Defined:** 2026-03-12
**Core Value:** AI learns to win (not draw) at Chénapan through AlphaZero-style self-play

## v1 Requirements

### Training Bug Fixes

- [x] **TRAIN-01**: Draw outcome returns -1 in `get_value_and_terminated()` — eliminates value head collapse where network learns to always predict ~0
- [x] **TRAIN-02**: Temperature applied to action probability sampling in `selfPlay()` — `temperature_action_probs` was computed but raw `action_probs` sampled (exploration was dead code)
- [x] **TRAIN-03**: Board hashing uses `hashlib.md5(state.tobytes())` instead of Python's session-randomized `hash()` — draw-by-repetition detection was silently broken across runs
- [x] **TRAIN-04**: Replay buffer batch slice uses `len(memory)` not `len(memory)-1` — last element was silently excluded from every training batch

### Training Pipeline

- [x] **PIPE-01**: `if __name__ == "__main__":` guard wraps training entry point — required so `gui.py` can import from `alpha_pan` without triggering a training run
- [x] **PIPE-02**: `tqdm.notebook` import replaced with plain `tqdm` — current import breaks when running as a standalone script outside Jupyter
- [x] **PIPE-03**: Per-iteration loss (policy + value) and game outcome (win/draw rate) logged to console during training — training is a black box without it
- [x] **PIPE-04**: Hyperparameter config block at top of file with real training values — current defaults (`num_iterations=3`, `num_selfPlay_iterations=1`) produce zero meaningful learning; scale to 50–200 iterations with 50–200 games per iteration

### Model Architecture

- [x] **MODEL-01**: `AlphaPanNet` rebuilt with residual blocks (AlphaZero-style skip connections) — enables deeper training without vanishing gradients; suitable for GPU training

### GUI

- [x] **GUI-01**: pygame-ce window renders 5×5 board with piece values visible for both players
- [x] **GUI-02**: Human player selects a piece then clicks destination to move; valid destinations highlighted after piece selection
- [x] **GUI-03**: AI move computed in background thread — prevents window freeze/"(Not Responding)" during MCTS search; required architectural decision from day one
- [x] **GUI-04**: Game-over screen displays outcome (win/loss/draw); pressing a key restarts the game without relaunching the program
- [x] **GUI-05**: Latest model checkpoint auto-detected and loaded on startup with `model.eval()` — ensures correct inference mode (BatchNorm/Dropout behave differently in training mode)

## v2 Requirements

### Training Observability

- **OBS-01**: CSV loss log written per iteration — enables post-hoc trend analysis across long runs
- **OBS-02**: Draw rate tracked per iteration — confirms draw=-1 is producing aggressive play over time

### GUI Polish

- **POLSH-01**: Last-move highlight — show which piece moved and where
- **POLSH-02**: "AI is thinking" animated indicator — visual feedback during MCTS computation
- **POLSH-03**: Configurable search count via GUI — slider or input to trade off play strength vs response time

## Out of Scope

| Feature | Reason |
|---------|--------|
| Online multiplayer | Out of scope — local human vs AI only |
| AI vs AI visualization | Not needed — training is headless |
| Move hints / engine analysis | Not needed — play mode only |
| Drag-and-drop piece input | Two-click is sufficient for a 5×5 board |
| Animation tweening | Scope creep — instant piece swap is fine |
| Neural network architecture experiments | MODEL-01 is the one change; further experiments are v2+ |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | Phase 1 | Complete |
| TRAIN-02 | Phase 1 | Complete |
| TRAIN-03 | Phase 1 | Complete |
| TRAIN-04 | Phase 1 | Complete |
| PIPE-01 | Phase 1 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| PIPE-04 | Phase 1 | Complete |
| MODEL-01 | Phase 1 | Complete |
| GUI-01 | Phase 2 | Complete |
| GUI-02 | Phase 2 | Complete |
| GUI-03 | Phase 2 | Complete |
| GUI-04 | Phase 2 | Complete |
| GUI-05 | Phase 2 | Complete |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after plan 02-01 completion (GUI-01, GUI-02 marked complete)*

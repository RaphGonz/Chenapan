# Technology Stack

**Analysis Date:** 2026-03-12

## Languages

**Primary:**
- Python 3.x - Game logic, neural network training, and MCTS implementation

## Runtime

**Environment:**
- Python interpreter (version inferred from code patterns as 3.6+)

**Package Manager:**
- pip (standard Python package manager)
- Lockfile: Not detected

## Frameworks

**Core:**
- PyTorch 1.x+ - Deep learning framework for neural networks
- NumPy 1.x+ - Numerical computing and array operations

**Utilities:**
- tqdm - Progress bar visualization (notebook variant `tqdm.notebook` used)
- random - Python standard library for stochastic operations
- math - Python standard library for mathematical functions

## Key Dependencies

**Critical:**
- `torch` - Core deep learning library for implementing AlphaPanNet neural network architecture
  - Includes: `torch.nn` (neural network modules), `torch.nn.functional` (loss functions)
  - Usage: Model training with BCE with logits and MSE loss functions
- `numpy` - Array manipulation and numerical operations
  - Usage: Board state representation, move generation, policy masks
- `tqdm` - Progress tracking during Jupyter notebook execution
  - Usage: `tqdm.notebook.trange` for training loop visualization

**Standard Library:**
- `random` - Random action selection during self-play and MCTS rollouts
- `math` - Mathematical operations in UCB calculation

## Configuration

**Environment:**
- No explicit environment configuration files detected
- GPU/CPU selection: Automatic detection via `torch.cuda.is_available()` (see `alpha_pan.py` line 840)

**Build:**
- No build configuration files detected
- Entry point: `alpha_pan.py` (single-file application)

## Training Configuration

**Hyperparameters:**
Located in `alpha_pan.py` lines 844-854:
- MCTS exploration constant: `C = 2`
- Number of MCTS searches per move: `num_searches = 60`
- Training iterations: `num_iterations = 3`
- Self-play games per iteration: `num_selfPlay_iterations = 1`
- Training epochs per iteration: `num_epochs = 1`
- Batch size: `batch_size = 64`
- Action temperature: `temperature = 1.25`
- Dirichlet noise epsilon: `dirichlet_epsilon = 0.1`
- Dirichlet noise alpha: `dirichlet_alpha = 0.3`

## Model Checkpoints

**Format:**
- PyTorch state dictionaries (`.pt` files)
- Model weights: `model_{iteration}.pt`
- Optimizer state: `optim_{iteration}.pt`
- Save location: Project root directory

**Storage:**
- Current implementation creates 3 checkpoints during default 3-iteration run:
  - `model_0.pt` (878 KB)
  - `model_1.pt` (878 KB)
  - `model_2.pt` (878 KB)
  - `optim_0.pt` (1.7 MB)
  - `optim_1.pt` (1.7 MB)
  - `optim_2.pt` (1.7 MB)

## Platform Requirements

**Development:**
- Python 3.6+ (based on f-string usage and type hints patterns)
- CUDA-compatible GPU (optional, falls back to CPU)
- 2+ GB RAM minimum for model training

**Production:**
- Python 3.6+ runtime
- PyTorch installation with CPU or GPU support
- Sufficient disk space for model checkpoints (3-5 MB per iteration)

---

*Stack analysis: 2026-03-12*

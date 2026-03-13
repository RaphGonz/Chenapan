# External Integrations

**Analysis Date:** 2026-03-12

## APIs & External Services

**Third-party APIs:**
- None detected - This is a self-contained game AI system with no external API calls

**Cloud Services:**
- Not integrated

## Data Storage

**Databases:**
- Not used - Game state exists only in memory during execution

**File Storage:**
- **Local filesystem only** - Model checkpoints and optimizer states saved as PyTorch `.pt` files
  - Location: Project root directory
  - Format: PyTorch state dictionaries
  - Files: `model_{iteration}.pt`, `optim_{iteration}.pt`

**Caching:**
- None - No persistent caching layer

**Temporary Storage:**
- Jupyter notebook checkpoints (`.ipynb_checkpoints/`)
- Purpose: Development environment snapshots

## Authentication & Identity

**Auth Provider:**
- Not applicable - No external authentication required
- All execution is local and standalone

## Monitoring & Observability

**Error Tracking:**
- Not integrated
- Current approach: Console output only

**Logs:**
- Console-based only (print statements)
- Examples:
  - Training progress: `print(f"this is the {selfPlay_iteration}th game")` (`alpha_pan.py` line 827)
  - Training epochs: `print(f"this is the {epochs}th epoch")` (`alpha_pan.py` line 833)
- PyTorch versions printed on startup: `print(np.__version__)`, `print(torch.__version__)` (`alpha_pan.py` lines 9, 14)

## CI/CD & Deployment

**Hosting:**
- None - Designed for local/development machine execution only

**CI Pipeline:**
- Not configured

**Version Control:**
- Not detected (no `.git` directory)

## Environment Configuration

**Required Environment Variables:**
- None - No external configuration required beyond installed dependencies

**Secrets Location:**
- Not applicable - No external credentials or API keys needed

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## System Integration Points

**GPU/Hardware:**
- PyTorch automatic device selection:
  - Code: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` (`alpha_pan.py` line 840)
  - Behavior: Falls back to CPU if CUDA is unavailable
  - No manual configuration required

**Jupyter Notebook Integration:**
- Uses `tqdm.notebook` for progress visualization
- Indicates project designed for interactive notebook environment
- Import: `from tqdm.notebook import trange` (`alpha_pan.py` line 20)

---

*Integration audit: 2026-03-12*

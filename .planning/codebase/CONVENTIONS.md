# Coding Conventions

**Analysis Date:** 2026-03-12

## Naming Patterns

**Files:**
- Single lowercase file with underscores: `alpha_pan.py`
- No test files present in codebase

**Classes:**
- PascalCase with descriptive names
- Examples: `Chenapan`, `AlphaPanNet`, `Node`, `MCTS`, `AlphaPan`
- Classes directly instantiated at module level (lines 839-857)

**Functions:**
- snake_case convention used throughout
- Examples: `flatten_and_sum_list_of_list()`, `get_hash()`, `get_initial_state()`, `swap()`, `check_ace_moves()`
- Descriptive, self-documenting names that indicate purpose (e.g., `check_king_moves()`, `get_valid_moves()`)

**Methods:**
- snake_case with leading underscore for private methods not observed (no private methods)
- Common prefixes: `get_*`, `check_*`, `is_*`, `update_*`
- Examples in `Chenapan` class (lines 84-389): `get_hash()`, `get_initial_state()`, `get_next_state()`, `get_valid_moves()`, `swap()`, `update_biggest_loop()`

**Variables:**
- snake_case for local and instance variables
- Descriptive names: `valid_moves`, `list_of_positions`, `number_of_moves`, `biggest_loop`, `start_row`, `action_probs`
- Loop counters: single letters (`i`, `j`) in nested loops
- Dictionary keys: lowercase strings (line 844-854): `'C'`, `'num_searches'`, `'batch_size'`, `'temperature'`

**Constants:**
- UPPERCASE with underscores
- Module-level constants (lines 24-71): `INITIAL_BOARD`, `VALID_MOVE_MATRIX`, `MAX_NUMBER_OF_MOVES`, `MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED`

**Types:**
- NumPy arrays: `state`, `policy`, `neutral_state`
- PyTorch tensors: prefixed with `out_` (line 805): `out_policy`, `out_value`
- No type hints observed throughout codebase

## Code Style

**Formatting:**
- No linter configuration file detected (no `.pylintrc`, `setup.cfg`, `.flake8`)
- No formatter configuration detected (no `.black`, `.autopep8`)
- Inconsistent spacing observed:
  - Some lines have trailing whitespace (visible in comments)
  - Indentation is 4 spaces (Python standard)
  - Blank lines between method definitions: 1-2 lines

**Line Length:**
- Generally under 100 characters
- Some longer lines up to ~120 characters (line 669, 753)

**Linting:**
- No linting tool configured
- Code contains intentional French comments indicating loose development style
- Comments like "Quelle horreur mais bon tant pis on va faire avec" (line 34) suggest this is a training/research project

## Import Organization

**Order:**
1. Standard library imports: `numpy`, `random`, `math` (lines 8-11)
2. Third-party deep learning: `torch`, `torch.nn`, `torch.nn.functional` (lines 13, 16-17)
3. Notebook imports: `tqdm.notebook` (line 20)

**Path Aliases:**
- `import numpy as np` - standard
- `import torch.nn as nn`
- `import torch.nn.functional as F`
- No custom path aliases or relative imports

## Error Handling

**Patterns:**
- Minimal error handling observed
- No try-except blocks in main code
- No validation of input parameters
- Relies on runtime errors for invalid operations (e.g., invalid actions)
- Comments indicate assumptions: "NORMALEMENT YA TOUJOURS DES MOVES DE DISPO" (line 569) - "NORMALLY THERE ARE ALWAYS MOVES AVAILABLE"

## Logging

**Framework:**
- `print()` statements only
- No logging library (no `logging` module)
- Direct console output for debugging (lines 9, 14, 486-492)

**Patterns:**
- Print statements for model input/output shapes (commented out, lines 486-492)
- Print statements in training loop (lines 827, 833): `print(f"this is the {selfPlay_iteration}th game")`
- Verbose output during training for iteration tracking

## Comments

**When to Comment:**
- Heavy use of inline comments for complex game logic
- French language comments dominate (research project, author's native language)
- Comments explain game mechanics and algorithm steps

**JSDoc/TSDoc:**
- Module-level docstring with author and date (lines 2-6)
- No function-level docstrings
- No type hints or signature documentation

**Comment Examples:**
- Algorithm explanation (lines 34-42): Comments explain piece values and valid move matrix
- Algorithm logic (lines 103-104, 118, 127-141): Inline comments for complex state management
- French comments throughout (lines 22, 35-39, 147-150, 195-197)

## Function Design

**Size:**
- Methods range from very short (2-3 lines) to long (60+ lines)
- Longest method: `check_ace_moves()` at 33 lines (193-226), involving repetitive move checking logic
- Shortest method: `get_opponent()` at 1 line (381)

**Parameters:**
- Methods typically 2-5 parameters
- Self parameter always present in class methods
- Optional parameters used: `update_meta_parameters = True` (line 116), `player = 1` (line 497)
- Dictionary passed as configuration: `self.args` used throughout (lines 500-501)

**Return Values:**
- Methods return single values or tuples
- Examples:
  - Single return: `return moves` (lists of valid moves)
  - Tuple return: `return 1, True` (value and termination status, line 370)
  - Tuple unpacking in callsites (line 367): `win, player = self.check_win(state, action)`

## Module Design

**Exports:**
- No explicit `__all__` definition
- Classes and functions automatically exported
- Module-level code at end (lines 839-857) creates instances and runs training

**Barrel Files:**
- Not applicable - single file structure

**Class Organization:**
- Game logic class: `Chenapan` (lines 84-389) - game rules and mechanics
- Neural network: `AlphaPanNet(nn.Module)` (lines 416-494) - PyTorch model
- Tree node: `Node` (lines 496-649) - Monte Carlo tree search node
- Search algorithm: `MCTS` (lines 651-726) - Monte Carlo tree search implementation
- Training loop: `AlphaPan` (lines 728-837) - self-play and training orchestration

**Class Relationships:**
- `AlphaPan` uses `MCTS`, `Chenapan`, `AlphaPanNet`
- `MCTS` uses `Node` and `Chenapan`
- `Node` uses `Chenapan` for game queries

---

*Convention analysis: 2026-03-12*

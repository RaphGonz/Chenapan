# Architecture

**Analysis Date:** 2026-03-12

## Pattern Overview

**Overall:** AlphaZero-inspired reinforcement learning system for the Chénapan board game, combining Monte Carlo Tree Search (MCTS) with deep neural networks for self-play training.

**Key Characteristics:**
- Game engine decoupled from AI training logic
- Neural network-guided MCTS replacing random rollouts
- Policy and value heads trained simultaneously on self-play data
- State encoding separates player perspective, opponent pieces, and game metadata
- Iterative self-play loops with batch training

## Layers

**Game Engine (`Chenapan` class):**
- Purpose: Implements core board game logic, move validation, and game state management
- Location: `alpha_pan.py` lines 84-390
- Contains: Board representation, move validation for piece types, win conditions, state transitions
- Depends on: NumPy for array operations
- Used by: Node (MCTS), AlphaPan (training), MCTS (search)

**Game State Encoding (`get_encoded_state`):**
- Purpose: Transforms raw board state into neural network-compatible format with 5 feature layers
- Location: `alpha_pan.py` lines 391-414
- Contains: Player layer, opponent layer, joker layer, move counter layer, repetition counter layer
- Depends on: Chenapan game engine for state interpretation
- Used by: AlphaPanNet (neural network input)

**Neural Network (`AlphaPanNet` class):**
- Purpose: Dual-head CNN that outputs policy (action probabilities) and value (position evaluation)
- Location: `alpha_pan.py` lines 416-494
- Contains: Compression block (conv layers), policy head (transpose convolutions), value head (dense layers)
- Depends on: PyTorch nn.Module, game state encoded as 5-channel tensors
- Used by: MCTS (search guidance), AlphaPan (training)

**Search Tree (`Node` class):**
- Purpose: Represents nodes in the MCTS tree with UCB-based selection and policy priors
- Location: `alpha_pan.py` lines 496-649
- Contains: Node state, children, visit counts, value sums, UCB calculation, tree expansion
- Depends on: Chenapan game engine for move validation and state transitions
- Used by: MCTS (tree traversal and expansion)

**Monte Carlo Tree Search (`MCTS` class):**
- Purpose: Guides game move selection using neural network-informed search
- Location: `alpha_pan.py` lines 651-726
- Contains: Root initialization, policy masking, UCB selection, expansion, backpropagation
- Depends on: AlphaPanNet (policy/value), Node (tree structure), Chenapan (game logic)
- Used by: AlphaPan (move selection during self-play)

**Training Controller (`AlphaPan` class):**
- Purpose: Orchestrates self-play generation and network training iterations
- Location: `alpha_pan.py` lines 728-837
- Contains: Self-play games, memory collection, batch training, model saving
- Depends on: MCTS, AlphaPanNet, Chenapan, PyTorch optimizer
- Used by: Main script execution

## Data Flow

**Self-Play Loop:**

1. `AlphaPan.learn()` initializes iteration loop (default 3 iterations)
2. Set model to eval mode
3. `AlphaPan.selfPlay()` generates games:
   - Initialize game state via `Chenapan.get_initial_state()`
   - For each player turn, call `MCTS.search(state)`:
     - Encode state with `Chenapan.get_encoded_state()`
     - Pass to `AlphaPanNet` for initial policy/value
     - Apply Dirichlet noise to exploration
     - Mask invalid moves using `Chenapan.get_valid_moves()`
     - Expand root node with masked policy via `Node.expand()`
     - Run 60 searches (default), selecting/expanding/backpropagating
   - Sample action from policy distribution
   - Update state via `Chenapan.get_next_state()` (with perspective change for player -1)
   - Check termination via `Chenapan.get_value_and_terminated()`
   - Store (encoded_state, action_probabilities, player) tuples in memory
   - If terminal, compute outcome and return formatted memory
4. Aggregate memory from all self-play iterations
5. Set model to train mode
6. Call `AlphaPan.train(memory)` for multiple epochs:
   - Shuffle memory
   - Process in batches
   - Forward pass: `AlphaPanNet(state)` returns (policy, value)
   - Compute losses: binary cross-entropy for policy, MSE for value
   - Backpropagate and optimize
7. Save model and optimizer checkpoints

**State Perspective Management:**

- Player 1 sees state as-is
- Player -1 sees state rotated 180° with all piece signs inverted via `Chenapan.change_perspective()`
- Before MCTS search for player -1: transform state to player 1 perspective
- After action selection: transform back to original perspective
- This ensures consistent network input orientation

**State Encoding (5 channels):**
1. Number of moves counter (scalar repeated across 5x5 grid)
2. Biggest loop counter (scalar repeated across 5x5 grid)
3. Opponent pieces layer (absolute values of negative pieces)
4. Joker layer (1 where piece equals 0, else 0)
5. Player pieces layer (positive piece values)

## Key Abstractions

**Valid Move Matrix (`VALID_MOVE_MATRIX`):**
- Purpose: Defines capture rules - which pieces can attack which other pieces by absolute value
- Examples: Piece 1 can capture [10,11,12], Piece 12 (King) can capture [2-12]
- Pattern: Each piece type has a list of pieces it can capture, enabling O(1) move validation

**Move Representation (action as tuple/list):**
- Format: `[start_position, end_position]` where position is 0-24 (row*5 + col)
- Decoded via: `row = position // 5`, `col = position % 5`
- Used throughout: move validation, state transitions, policy/value mapping

**Piece Encoding (signed integers):**
- Positive: Player 1 pieces (1-12, where 0=joker)
- Negative: Player -1 pieces (-1 to -12)
- Absolute value determines piece type: 1=Ace, 2-9=numbered pieces, 10=Jack, 11=Queen, 12=King

**Policy Format (25x25 matrix):**
- Rows: source positions (0-24)
- Columns: destination positions (0-24)
- Values: probability of selecting that move
- Masked to zero for invalid moves before softmax and action sampling

## Entry Points

**Script Execution (`alpha_pan.py` lines 839-857):**
- Location: `alpha_pan.py` bottom section
- Triggers: Direct Python execution
- Responsibilities: Initialize game, model, optimizer, args, create AlphaPan trainer, call `learn()`
- Device: CUDA if available, else CPU

**Game Initialization (`Chenapan.__init__`):**
- Location: `alpha_pan.py` lines 85-96
- Triggers: Called once at startup
- Responsibilities: Set board dimensions (5x5), action space (25x25), reset game state tracking

**Network Initialization (`AlphaPanNet.__init__`):**
- Location: `alpha_pan.py` lines 417-483
- Triggers: Called once at startup
- Responsibilities: Build compression block, policy head, value head; move to device

## Error Handling

**Strategy:** Minimal explicit error handling; relies on NumPy/PyTorch exceptions

**Patterns:**
- Boundary checking in move validation (row/column >= 0 and <= 4)
- Game termination on invalid state (no moves, exceeded move limit, position repetition 3x)
- Implicit tensor shape validation via PyTorch forward pass

## Cross-Cutting Concerns

**Logging:** None - uses print statements for iteration progress (lines 827, 833)

**Validation:**
- Move validation in piece-specific methods (`check_ace_moves`, `check_jack_moves`, etc.)
- Policy masking to enforce valid moves
- Win condition verification via absolute value of piece at terminal position

**State Perspective Transformation:** Centralized in `Chenapan.change_perspective()` - rotates state 180° and inverts signs for player -1

---

*Architecture analysis: 2026-03-12*

# Codebase Concerns

**Analysis Date:** 2026-03-12

## Tech Debt

**Hardcoded Board Constants:**
- Issue: Game logic relies entirely on hardcoded constants (INITIAL_BOARD, VALID_MOVE_MATRIX) making the game rules inflexible and impossible to modify or extend
- Files: `alpha_pan.py` (lines 24-71)
- Impact: Any rule changes require modifying the source code. The VALID_MOVE_MATRIX is manually maintained with implicit rules about piece interactions that are not documented
- Fix approach: Extract board configuration to a configuration class or data structure. Document the rule hierarchy (which pieces can attack which) in a separate configuration module

**Hash-Based State Tracking Using Python hash():**
- Issue: Uses Python's built-in `hash()` function for board state tracking instead of cryptographic or deterministic hashing
- Files: `alpha_pan.py` (line 99: `get_hash()` method)
- Impact: Python's `hash()` has per-session randomization by default (PYTHONHASHSEED). This means state comparison across sessions may fail. The `biggest_loop` mechanism for detecting repeated positions is unreliable
- Fix approach: Use `hashlib.md5()` or `hashlib.sha256()` for deterministic hashing. Alternatively, use numpy array serialization with a consistent hashing approach

**Inefficient Position Hashing Strategy:**
- Issue: Converting entire board state to string then hashing is expensive. For MCTS with 60+ searches per move, this is a performance bottleneck
- Files: `alpha_pan.py` (lines 98-99, 128-129)
- Impact: Each move requires O(n) string conversion and hashing operations during tree search. With high search counts (num_searches=60), this accumulates quickly
- Fix approach: Use tuple hashing of flattened board or implement zobrist hashing for O(1) state incremental updates

**Manual Move Validation with Repetitive Conditionals:**
- Issue: Move checking logic is extremely repetitive with nearly identical code patterns duplicated across five methods
- Files: `alpha_pan.py` (lines 193-332: check_ace_moves, check_jack_moves, check_queen_moves, check_king_moves, check_basic_moves)
- Impact: Code is difficult to maintain and debug. The 8-directional movement patterns in check_ace_moves is repeated in check_king_moves with different ranges. Changes to move logic require updates in multiple places
- Fix approach: Create a generic move generator function that takes a piece type and direction patterns as parameters, eliminating code duplication

**Inconsistent State Mutation Side Effects:**
- Issue: The `get_next_state()` method modifies state in-place but also has a flag to control whether it updates metadata (number_of_moves, biggest_loop, list_of_positions)
- Files: `alpha_pan.py` (lines 116-144)
- Impact: Hard to reason about when global game state is being modified. The `update_meta_parameters` flag creates two distinct code paths that are easy to misuse
- Fix approach: Separate concerns - return new state without side effects. Store metadata updates in a separate tracking object or handle them at the caller level

**Comments in French with Poor English Translation:**
- Issue: Source code comments are in French with English variable names, making code harder to navigate for non-French speakers
- Files: `alpha_pan.py` (throughout, examples: lines 34-35, 147-149, 197-198)
- Impact: Reduces code accessibility and maintainability in collaborative environments
- Fix approach: Standardize on English for all code comments and docstrings

## Known Bugs

**Off-by-One Error in Batch Processing:**
- Symptoms: Training batches may skip or duplicate the last element in the memory buffer
- Files: `alpha_pan.py` (line 797)
- Trigger: When processing memory batches: `sample = memory[batchIndex:min(len(memory)-1,batchIndex + self.args["batch_size"])]`
- Problem: Using `len(memory)-1` instead of `len(memory)` causes the final element to never be included in any batch. The last batch will be smaller than intended
- Workaround: None - data loss during training
- Fix: Change to `memory[batchIndex:min(len(memory), batchIndex + self.args["batch_size"])]`

**Unused simulate() Method:**
- Symptoms: Classic MCTS rollout simulation code is commented out but method still exists
- Files: `alpha_pan.py` (lines 608-642, comment at line 711)
- Problem: Dead code creates confusion about intended algorithm. The method is never called in AlphaPan.learn() because AlphaPan uses neural network policy evaluation instead
- Fix approach: Remove the `simulate()` method entirely or document why it's retained

**Missing Return Statement in expand():**
- Symptoms: Node.expand() creates children but returns None, unlike typical tree expansion patterns
- Files: `alpha_pan.py` (lines 579-606)
- Problem: Method has no return value despite modifying tree structure. Inconsistent with object-oriented patterns
- Impact: Minor - doesn't break functionality since children are added to self.children, but violates principle of least surprise
- Fix: Return the expanded node or list of children for consistency

**Perspective Change Not Applied to Value Target Labels:**
- Symptoms: During self-play, value targets are stored from player's perspective but might not correctly account for perspective switches
- Files: `alpha_pan.py` (lines 781-787)
- Problem: When player=-1, the neutral state is obtained via `change_perspective()`, but the value labeling logic at lines 783 needs verification that `player` variable matches the actual perspective
- Impact: Subtle training bias if perspective handling is incorrect
- Risk level: Medium - requires careful trace-through of perspective switches to confirm

## Security Considerations

**No Input Validation on Action Selection:**
- Risk: Random action selection uses `np.random.choice()` without bounds validation
- Files: `alpha_pan.py` (lines 755, 769)
- Current mitigation: Valid moves are pre-filtered through MCTS search, so invalid actions are unlikely but not impossible
- Recommendations: Add assertions to validate that selected actions exist in the valid moves list before applying state change

**Model Deserialization Without Validation:**
- Risk: If trained models are loaded from untrusted sources, no verification of model integrity occurs
- Files: `alpha_pan.py` (lines 836-837 save, but no load code shown)
- Current mitigation: None visible in provided code
- Recommendations: Add model checksum validation or size bounds checking before loading .pt files

**No Seed Management for Reproducibility:**
- Risk: `np.random.choice()` and `random.shuffle()` use unseeded randomness, making runs non-reproducible
- Files: `alpha_pan.py` (lines 795, 755, 769)
- Current mitigation: None
- Recommendations: Add a seed parameter to AlphaPan and use `np.random.seed()` and `random.seed()` for reproducible training

## Performance Bottlenecks

**State Hashing in Hot Loop:**
- Problem: Every MCTS action creates a hash that requires string conversion of entire 5x5 board
- Files: `alpha_pan.py` (lines 128-130 in get_next_state, called 60+ times per move)
- Cause: Naive string conversion approach: `hash(str(state))`
- Improvement path: Implement zobrist hashing or use numpy array hashing; cache hash values in nodes

**Repeated get_valid_moves() Calls:**
- Problem: Valid moves are calculated multiple times for the same state (lines 671, 698)
- Files: `alpha_pan.py` (lines 671, 698)
- Cause: No caching mechanism for move generation results
- Current impact: With 5x5=25 positions and 8 directions each, this is computationally expensive; multiplied by num_searches=60 per move
- Improvement path: Cache valid moves by state hash or add move generation memoization

**Neural Network Inference Per Search Iteration:**
- Problem: The model is called twice per MCTS iteration (root expansion, node expansion) without batching
- Files: `alpha_pan.py` (lines 663-666, 691-694)
- Cause: Individual forward passes instead of batch inference
- Impact: GPU utilization is poor; CPU-GPU transfer overhead dominates
- Improvement path: Batch multiple MCTS searches together or implement asynchronous parallel MCTS

**Inefficient list.count() in Loop:**
- Problem: `update_biggest_loop()` calls `list.count(h)` which is O(n) on the history list
- Files: `alpha_pan.py` (line 108)
- Cause: Called after every move in get_next_state
- Impact: O(n) behavior accumulates as game progresses; by 50 moves, this is 50*50=2500 list searches
- Improvement path: Maintain a dictionary counter instead of using list.count()

## Fragile Areas

**Board Coordinate Transformation Logic:**
- Files: `alpha_pan.py` (lines 119-122, 351-354, throughout move checking functions)
- Why fragile: Converts between 2D (row,col) and 1D flat indices using hardcoded math (row*5 + col). The magic number 5 appears throughout
- Safe modification: Extract to constants ROW_COUNT=5, COLUMN_COUNT=5. Create helper functions: `to_flat_index(row, col)`, `from_flat_index(idx)`
- Test coverage: Move validation functions lack unit tests; changes could silently break move generation

**VALID_MOVE_MATRIX Maintenance:**
- Files: `alpha_pan.py` (lines 44-71)
- Why fragile: This 25-element list encodes game rules but has no documentation of what each index means beyond comments
- Safe modification: Convert to a configuration object with named attributes. Add comprehensive docstring explaining the piece hierarchy
- Test coverage: No validation that pieces can only capture pieces they're supposed to

**State Encoding for Neural Network:**
- Files: `alpha_pan.py` (lines 391-414)
- Why fragile: `get_encoded_state()` creates 5 layers (number_of_moves, biggest_loop, adverse, joker, player) with hardcoded indices
- Safe modification: Create an Enum for layer indices. Document the meaning of each layer
- Test coverage: No unit tests verify that encoding produces expected shapes or values

**Memory Management in Self-Play:**
- Files: `alpha_pan.py` (lines 737-788)
- Why fragile: Large memory lists accumulate without size limits. After num_selfPlay_iterations games, all game states are retained
- Safe modification: Add memory pruning or use generators. Set max memory size with overflow policy
- Test coverage: No tests for behavior with large memory buffers

## Scaling Limits

**Memory Growth with Self-Play:**
- Current capacity: Stores every board state from every self-play game in memory
- Limit: With 5x5 boards (50 moves max per game) × num_selfPlay_iterations=1 × num_iterations=3, manageable. But scales to 10+ games per iteration, memory becomes problematic
- Scaling path: Implement experience replay buffer with fixed size. Use online learning or mini-batch updates

**Model Size and Inference:**
- Current capacity: Model is ~858KB (trained), optimizer state ~1.7MB
- Limit: Current architecture uses ConvTranspose2d for policy upsampling from 1x1 to 25x25, which is unusual and may not scale to larger boards
- Scaling path: If extending to larger boards, replace deconvolution with reshape-based approaches or attention mechanisms

**Single-GPU Training:**
- Current capacity: No distributed training support
- Limit: Training bottlenecks at MCTS search and model inference; single GPU cannot parallelize independent games
- Scaling path: Implement async self-play workers with centralized model server (AlphaGo Zero pattern)

## Dependencies at Risk

**No Python Version Constraint:**
- Risk: `alpha_pan.py` has no `requirements.txt` or Python version specification
- Impact: Unknown versions of numpy, torch, tqdm create reproducibility issues and potential breaking changes
- Migration plan: Create `requirements.txt` with pinned versions: numpy==1.24.x, torch==2.x, tqdm==4.x

**Hardcoded Version Printing:**
- Risk: Lines 9 and 14 print library versions to stdout; no version compatibility checks
- Files: `alpha_pan.py` (lines 9, 14)
- Impact: Version mismatches silently fail or produce unexpected behavior
- Migration plan: Add version assertions at startup; fail fast if requirements not met

## Missing Critical Features

**No Training Checkpointing or Recovery:**
- Problem: If training crashes, all progress is lost (or must load last saved iteration)
- Files: `alpha_pan.py` (lines 836-837)
- Blocks: Long training runs are risky; cannot pause and resume gracefully
- Fix: Implement checkpoint saving with iteration metadata; support resuming from any checkpoint

**No Hyperparameter Configuration File:**
- Problem: All hyperparameters hardcoded in script (lines 844-854)
- Files: `alpha_pan.py` (args dictionary)
- Blocks: Cannot easily run multiple experiments with different parameters
- Fix: Move args to YAML/JSON config file; load at startup

**No Evaluation Against Baseline or Self-Play Metrics:**
- Problem: No win rate tracking, ELO rating, or performance metrics saved
- Blocks: Cannot determine if training is improving or just memorizing
- Fix: Add game outcome tracking; compute win percentage over time

**No Inference Interface:**
- Problem: No way to load trained model and play games without the training loop
- Blocks: Cannot evaluate trained models or play against them
- Fix: Add separate inference script that loads model and runs MCTS without training

## Test Coverage Gaps

**No Unit Tests for Move Generation:**
- What's not tested: The five move-checking functions (check_ace_moves, check_jack_moves, etc.) have no validation
- Files: `alpha_pan.py` (lines 193-332)
- Risk: Moving a piece could produce invalid destinations without detection
- Priority: High - move validation is critical to game correctness

**No Tests for State Hash Consistency:**
- What's not tested: Whether hash() produces consistent results across game states
- Files: `alpha_pan.py` (line 99)
- Risk: Repeat position detection could fail silently
- Priority: High - game termination rules depend on this

**No Tests for Board Perspective Change:**
- What's not tested: `change_perspective()` rotates board and flips signs correctly
- Files: `alpha_pan.py` (line 386)
- Risk: Player -1 MCTS could see incorrect board state
- Priority: High - affects training correctness

**No Tests for MCTS Selection/Expansion:**
- What's not tested: UCB calculation, node selection, policy masking
- Files: `alpha_pan.py` (lines 532-710)
- Risk: Invalid moves could be selected if masking fails
- Priority: Medium - foundational to tree search correctness

**No Tests for Training Loop:**
- What's not tested: Batch processing, loss computation, gradient updates
- Files: `alpha_pan.py` (lines 794-817)
- Risk: Silent training failures; batch slicing bug (line 797) would not be caught
- Priority: Medium - off-by-one error in batch processing

**No Integration Tests:**
- What's not tested: Full game-play flow from reset → moves → termination
- Files: `alpha_pan.py` (entire learn loop)
- Risk: Multiple bugs could combine to produce invalid games
- Priority: Medium - verifies game mechanics work end-to-end

---

*Concerns audit: 2026-03-12*

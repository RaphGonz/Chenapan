# Testing Patterns

**Analysis Date:** 2026-03-12

## Test Framework

**Status:** Not detected

**Current State:**
- No test files found in codebase
- No test framework imported (pytest, unittest, nose, etc.)
- No test configuration files present (.pytest.ini, setup.cfg, pyproject.toml)
- No test runner commands in documentation

**Testing Approach:**
- Manual/interactive testing indicated by commented-out code (lines 859-931)
- Interactive game loop commented out showing manual gameplay testing approach

## Code Organization for Testing

**Location:**
- No separate test directory
- No test files alongside source code
- Commented code block (lines 859-931) shows manual test flow:
  ```python
  # chenapan = Chenapan()
  # player = 1
  # args = {
  #     'C': 2,
  #     'num_searches': 100
  # }
  # model = AlphaPanNet()
  # model.eval()
  # mcts = MCTS(chenapan,args,model)
  # state = chenapan.get_initial_state()
  # while True:
  #     print(state)
  ```

**Naming:**
- Test files not applicable
- Code follows `alpha_pan.py` naming convention

## Testing Strategy

**Manual Testing:**
- Interactive game loop for manual play-testing (commented code, lines 875-930)
- Print-based debugging for algorithm verification
- Model evaluation mode used: `model.eval()` (line 824, 869)

**Training Loop Testing:**
- Self-play validation (line 737-788): `selfPlay()` method generates training data
- Training iteration logging (lines 827, 833):
  ```python
  print(f"this is the {selfPlay_iteration}th game")
  print(f"this is the {epochs}th epoch")
  ```

**Current Debug Output:**
- Model input/output shape verification (lines 486-492, commented):
  ```python
  # print("Input : ",x.shape)
  # print("Encoded : ", x.shape)
  # print("Policy shape : ",policy.shape)
  # print("Value shape : ",value.shape)
  ```

## Model Validation Patterns

**No Explicit Test Assertions:**
- No assertions present in code
- No validation of output ranges
- Loss values computed but not validated (lines 810-813):
  ```python
  policy_loss = F.binary_cross_entropy_with_logits(out_policy,policy_targets)
  value_loss = F.mse_loss(out_value,value_targets)
  loss = policy_loss + value_loss
  ```

**Model Checkpointing:**
- State dict saved after each training iteration (lines 836-837):
  ```python
  torch.save(self.model.state_dict(),f"model_{iteration}.pt")
  torch.save(self.optimizer.state_dict(),f"optim_{iteration}.pt")
  ```

## Data Validation

**Input Validation:**
- Minimal validation in `get_next_state()` (lines 116-144)
- No explicit checks for invalid actions
- Assumes valid state representation

**Game Logic Verification:**
- Move validation in `check_*_moves()` methods (lines 193-332)
- Uses `VALID_MOVE_MATRIX` lookup for piece interaction validation
- Boundary checks in move generation (lines 201-212 in `check_ace_moves()`)

**Example Boundary Check Pattern:**
```python
if row-i >= 0:
    if abs(state[row-i,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
        moves.append((row-i)*5 + column)
```

## PyTorch/Neural Network Testing

**Model Shape Verification:**
- Commented debug output shows shape checking (lines 807-808):
  ```python
  if out_policy.dim() == 4:
      out_policy = out_policy.squeeze(1)  # devient (B, 25, 25)
  ```

**Device Placement:**
- Device explicitly set in model initialization (line 420)
- Tensor device transfer verified in training (line 801-803):
  ```python
  state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
  policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
  value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
  ```

**Gradient Computation:**
- `@torch.no_grad()` decorator used in search to prevent gradient accumulation (line 657)
- Standard backward pass in training (line 816):
  ```python
  loss.backward()
  self.optimizer.step()
  ```

## Coverage

**Requirements:** None enforced

**What's Untested:**
- Game termination conditions (lines 365-378 `get_value_and_terminated()`)
- Perspective change correctness (line 386-389 `change_perspective()`)
- Monte Carlo tree search selection logic (lines 532-550)
- Policy masking and normalization (lines 700-705)
- State encoding for network input (lines 391-414)
- Complex move validation for edge pieces

## Recommendations for Testing

**Unit Testing Candidates:**
- `Chenapan.get_valid_moves()` - Should validate move generation for each piece type
- `Chenapan.check_win()` - Should test terminal state detection
- `AlphaPanNet.forward()` - Should verify output shapes match expected (25, 25) for policy and (1,) for value
- `Node.get_ucb()` - Should test UCB calculation correctness

**Integration Testing Candidates:**
- Full game simulation from `selfPlay()` - Verify game reaches termination
- Training loop from `learn()` - Verify model updates after each epoch
- MCTS search convergence - Verify tree expands correctly

**Test Framework Suggestion:**
- pytest would be suitable for unit tests
- Simple fixture structure for game state initialization
- Parametrized tests for different piece movements

---

*Testing analysis: 2026-03-12*

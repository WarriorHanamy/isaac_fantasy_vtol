# Control Mode Documentation

## Current Configuration

The drone racer environment is currently configured for **"body_rate" control mode**, not "motor" mode.

### Configuration Details

**File**: `tasks/drone_racer/drone_racer_env_cfg.py:74`
```python
control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)
```

**File**: `tasks/drone_racer/mdp/actions.py:209`
```python
control_mode: str = "body_rate"  # Default value
```

## Control Modes

### 1. "body_rate" Mode (Current)
- **Actions**: `[throttle, rate_x, rate_y, rate_z]`
- **Processing**:
  - `throttle_cmd = clamped[:, 0]`
  - `desired_rates = clamped[:, 1:] * self._max_body_rate`
  - Uses body-rate controller to compute torque commands
  - Converts wrench commands to motor speeds via allocation matrix

### 2. "motor" Mode
- **Actions**: Direct motor commands `[motor1, motor2, motor3, motor4]`
- **Processing**:
  - `mapped = (clamped + 1.0) / 2.0`
  - `omega_ref = self.cfg.omega_max * mapped`
  - No body-rate controller involved

## Logging Behavior

### In "body_rate" Mode:
- `throttle_cmd` → logged as `"throttle_cmd"`
- `rate_cmd_x`, `rate_cmd_y`, `rate_cmd_z` → logged as `"rate_cmd_x/y/z"`
- `torque_cmd_x`, `torque_cmd_y`, `torque_cmd_z` → logged as `"torque_cmd_x/y/z"`
- `a1`, `a2`, `a3`, `a4` → raw actions (always logged)
- `w1`, `w2`, `w3`, `w4` → motor speeds (always logged)

### In "motor" Mode:
- Only `a1-a4` and `w1-w4` are logged
- No intermediate control signals are available

## Troubleshooting TensorBoard

If you don't see `throttle_cmd` etc. in TensorBoard while in "body_rate" mode:

1. **Check Logger Configuration**: Ensure your logger setup includes these terms
2. **Verify Run Selection**: Make sure you're looking at the correct TensorBoard run
3. **Check Filtering**: Ensure no filters are hiding these metrics
4. **Confirm Mode**: Verify you're actually in "body_rate" mode during training

## Switching Modes

To switch to "motor" mode, modify the configuration:
```python
control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(
    use_motor_model=True,  # or control_mode="motor"
    control_mode="motor"
)
```
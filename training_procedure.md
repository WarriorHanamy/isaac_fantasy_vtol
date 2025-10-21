# Drone Racing Training Procedure and Dynamics Analysis
## Action space
A Benchmark Comparison of Learned Control Policies  for Agile Quadrotor Flight: into a) Linear Velocity Commands (LV), b) Collective Thrust and Bodyrates (CTBR), and c) Single Rotor Thrusts (SRT).

## Overview

This document outlines the training procedure for drone racing using the IsaacLab framework, focusing on how the dynamics modules affect the training pipeline. The system implements a reinforcement learning approach for autonomous drone racing through gates.

## Dynamics Modules Analysis

### 1. Motor Dynamics (`dynamics/motor.py`)**



**Purpose**: Simulates realistic motor behavior with first-order dynamics and rate constraints.

**Key Features**:
- **First-order motor dynamics**: `ω_rate = (1/τ) * (ω_ref - ω)`
- **Rate limiting**: Constrains motor acceleration/deceleration to realistic values
- **Configurable per-motor parameters**: Time constants, initial velocities, rate limits
- **Optional bypass**: `use_motor_model` flag can disable dynamics for faster training  (#### Could be used for two-staged training, disable, it will behave like a planning.)

**Training Impact**:
- **Realism**: Introduces motor lag and inertia, making control more challenging
- **Learning difficulty**: Agent must account for motor response delays
- **Safety**: Prevents unrealistic instantaneous motor changes
- **Training speed**: When disabled (`use_motor_model=False`), training is faster but less realistic

### 2. Control Allocation (`dynamics/allocation.py`)

**Purpose**: Converts between motor forces and body wrenches (thrust + torques) using the control allocation matrix.

**Key Features**:
- **Quadrotor allocation matrix**: Maps 4 motor thrusts to total thrust and 3 torques
- **Inverse allocation**: Converts desired body wrenches to individual motor commands
- **Thrust coefficient mapping**: `thrust = thrust_coeff * ω²`
- **Clamping**: Ensures motor thrusts stay within physical limits

**Training Impact**:
- **Control decoupling**: Allows agent to think in body rates rather than direct motor commands
- **Physical constraints**: Enforces quadrotor actuation limits
- **Numerical stability**: Uses pseudo-inverse for robust computation

### 3. Body Rate Controller (`dynamics/controllers.py`)

**Purpose**: Implements a Lee-rate style body rate controller with gyroscopic feedforward compensation.

**Key Features**:
- **PD control**: Rate error feedback with configurable gains
- **Gyroscopic compensation**: `ω × (I × ω)` feedforward term
- **Rate limiting**: Enforces maximum body angular rates
- **Inertia-aware**: Accounts for drone inertia properties

**Training Impact**:
- **Simplification**: Agent can command desired rates instead of raw torques
- **Stability**: Provides stable low-level control
- **Realism**: Includes gyroscopic effects present in real quadrotors

## Training Pipeline and Calling Stacks

### 1. Environment Initialization
```
train.py → DroneRacerEnvCfg → ControlActionCfg
```

**Flow**:
1. `train.py` creates environment with `DroneRacerEnvCfg`
2. Environment initializes `ControlAction` with dynamics modules
3. Dynamics modules are instantiated with configured parameters

### 2. Action Processing Pipeline
```
Agent Actions → ControlAction.process_actions() → Dynamics Chain → Robot Forces
```

**Detailed Flow**:
1. **Raw Actions** (`[-1, 1]`): Neural network outputs
2. **Action Mapping**: Scaled to appropriate control mode
   - **Motor mode**: Direct motor velocity commands
   - **Body rate mode**: Throttle + angular rate commands
3. **Control Loop**:
   - **Body Rate Controller**: `torque_cmd = BodyRateController.compute(current_rates, desired_rates)`
   - **Allocation**: `omega_ref = Allocation.omega_from_wrench(wrench_cmd)`
   - **Motor Dynamics**: `omega_real = Motor.compute(omega_cmd)`
   - **Force Computation**: `thrust_torque = Allocation.compute(omega_real)`
4. **Physics Application**: Forces applied to robot in simulation

### 3. Environment Step Cycle
```
Environment.step() →
  Agent.act() →
  ControlAction.process_actions() →
  ControlAction.apply_actions() →
  Physics simulation →
  Observations/Rewards
```

## Control Modes and Their Impact

### 1. Direct Motor Control (`control_mode="motor"`)
- **Actions**: 4 normalized values mapped to motor velocities
- **Complexity**: High - agent must learn motor mixing
- **Realism**: Maximum
- **Training difficulty**: Highest

### 2. Body Rate Control (`control_mode="body_rate"`)
- **Actions**: `[throttle, rate_x, rate_y, rate_z]`
- **Complexity**: Medium - allocation and rate control handled by dynamics
- **Realism**: High with motor dynamics enabled
- **Training difficulty**: Moderate (default configuration)

## Monitoring Training Metrics

### TensorBoard (Losses & Episodic Rewards)
- **Where**: Training automatically writes TensorBoard event files to `logs/skrl/<experiment>/`.
- **Enable finer logging**: Override the writer interval from the CLI, for example  
  `python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 agent.experiment.write_interval=1000`.
- **View during/after training**: In another terminal run `tensorboard --logdir logs/skrl --port 6006` and open <http://localhost:6006>.  
  Key scalars include `policy_loss`, `value_loss`, `entropy`, `learning_rate`, and `rollout/episodic_return`.

### Reward Components
- **What**: Each reward term now logs under `reward/<term_name>` (e.g., `reward/progress`, `reward/gate_passed`).
- **During evaluation**: Use the CSV logger in `play.py`, for example  
  `python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --log 5`  
  to record 5 evaluation episodes into `logs/skrl/.../log_<timestamp>.csv`.
- **Training insight**: Combine the CSV reward traces with TensorBoard’s episodic returns to understand which terms are dominating learning across epochs.

## Configuration Parameters and Their Effects

### Motor Parameters
- **`use_motor_model`**: Toggle for motor dynamics (default: False for faster training)
- **`taus`**: Motor time constants (smaller = faster response)
- **`max_rate/min_rate`**: Motor acceleration limits
- **`omega_max`**: Maximum motor velocity

### Control Parameters
- **`rate_gains`**: Body rate controller gains (affects stability)
- **`max_body_rate`**: Angular rate limits
- **`throttle_limits`**: Thrust constraints
- **`body_inertia`**: Inertial properties for feedforward compensation

### Physical Parameters
- **`arm_length`**: Quadrotor geometry (affects control allocation)
- **`thrust_coef`**: Motor thrust constant
- **`drag_coef`**: Motor drag coefficient

## Training Considerations

### 1. Realism vs. Training Speed
- **Motor dynamics disabled**: Faster training, less realistic behavior
- **Motor dynamics enabled**: Slower training, more transferable policies

### 2. Control Mode Selection
- **Beginner training**: Start with body rate control
- **Advanced training**: Consider direct motor control for fine-grained optimization

### 3. Parameter Tuning
- **Rate gains**: Balance between responsiveness and stability
- **Motor limits**: Must match hardware specifications
- **Allocation parameters**: Should match drone geometry

## Debugging and Logging

The system includes comprehensive logging at `tasks/drone_racer/mdp/actions.py:139-147`:
- Raw actions
- Motor velocities
- Thrust and torque commands
- Rate commands and actual rates

## File Structure
```
dynamics/
├── __init__.py          # Module exports
├── motor.py            # Motor dynamics simulation
├── allocation.py       # Control allocation matrix
└── controllers.py      # Body rate controller

tasks/drone_racer/
├── mdp/actions.py      # Action processing with dynamics integration
└── drone_racer_env_cfg.py  # Environment configuration

scripts/rl/
└── train.py           # Main training script
```

This architecture provides a modular and realistic simulation environment for training autonomous drone racing policies while maintaining the flexibility to adjust realism for training efficiency.

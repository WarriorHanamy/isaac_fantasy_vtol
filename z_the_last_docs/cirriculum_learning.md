# Curriculum Learning for Drone Racing

This document explains how to implement curriculum learning for drone racing by dynamically adjusting the drone's collision area during training.

## Overview

Curriculum learning gradually increases task difficulty by modifying the drone's collision geometry:
- **Easy**: Larger collision area (more forgiving)
- **Medium**: Slightly larger collision area
- **Hard**: Normal collision area
- **Expert**: Smaller collision area (more challenging)

## Available Configurations

The `assets/five_in_drone.py` file provides several configurations for curriculum learning:

### Predefined Configurations
```python
from assets.five_in_drone import (
    FIVE_IN_DRONE,      # Default configuration
    EASY_DRONE_CFG,     # 20% larger collision area
    MEDIUM_DRONE_CFG,   # 10% larger collision area
    HARD_DRONE_CFG,     # Original collision area
    EXPERT_DRONE_CFG    # 10% smaller collision area
)
```

### Dynamic Configuration Function
```python
from assets.five_in_drone import create_drone_cfg_with_collision_scale

# Create custom collision scaling (1.0 = original, >1.0 = larger, <1.0 = smaller)
custom_cfg = create_drone_cfg_with_collision_scale(1.15)  # 15% larger
```

## Usage in Training

### Method 1: Predefined Stages

Use predefined configurations based on training stages:

```python
def get_curriculum_cfg(curriculum_stage):
    """Get drone configuration based on curriculum stage."""
    from assets.five_in_drone import (
        EASY_DRONE_CFG, MEDIUM_DRONE_CFG,
        HARD_DRONE_CFG, EXPERT_DRONE_CFG
    )

    if curriculum_stage == "easy":
        return EASY_DRONE_CFG
    elif curriculum_stage == "medium":
        return MEDIUM_DRONE_CFG
    elif curriculum_stage == "hard":
        return HARD_DRONE_CFG
    elif curriculum_stage == "expert":
        return EXPERT_DRONE_CFG
    else:
        return HARD_DRONE_CFG  # default

# In your environment setup
def setup_env(curriculum_stage="hard"):
    drone_cfg = get_curriculum_cfg(curriculum_stage)
    # Pass drone_cfg to your environment constructor
    env = DroneRacingEnv(robot_cfg=drone_cfg)
    return env
```

### Method 2: Performance-Based Adaptation

Adjust collision area based on training performance:

```python
def get_adaptive_cfg(success_rate, episode_count):
    """Dynamically adjust collision area based on performance."""
    from assets.five_in_drone import create_drone_cfg_with_collision_scale

    # Scale from 1.3 (easy) to 0.9 (expert) based on success rate
    # success_rate: 0.0 to 1.0
    scale = 1.3 - (success_rate * 0.4)

    # Add episode count consideration for gradual progression
    episode_factor = min(episode_count / 10000, 1.0)  # Cap at 10k episodes
    scale = scale * (1.0 - episode_factor * 0.15) + 0.9 * episode_factor

    return create_drone_cfg_with_collision_scale(scale)

# In your training loop
def train():
    env = DroneRacingEnv()
    episode_count = 0

    for epoch in range(num_epochs):
        success_rate = calculate_success_rate()

        # Update drone configuration based on performance
        new_cfg = get_adaptive_cfg(success_rate, episode_count)
        env.update_robot_config(new_cfg)

        # Training logic here...
        episode_count += episodes_per_epoch
```

### Method 3: Step-Based Progression

Gradually reduce collision area over training steps:

```python
def get_step_based_cfg(total_steps):
    """Get configuration based on total training steps."""
    from assets.five_in_drone import create_drone_cfg_with_collision_scale

    # Define curriculum milestones
    curriculum_schedule = [
        (0, 1.3),      # Start: 30% larger collision
        (2000, 1.2),   # After 2k steps: 20% larger
        (5000, 1.1),   # After 5k steps: 10% larger
        (10000, 1.05), # After 10k steps: 5% larger
        (15000, 1.0),  # After 15k steps: normal
        (20000, 0.95), # After 20k steps: 5% smaller
        (25000, 0.9)   # After 25k steps: 10% smaller
    ]

    # Find appropriate scale for current step
    scale = 1.0  # default
    for step_threshold, step_scale in curriculum_schedule:
        if total_steps >= step_threshold:
            scale = step_scale
        else:
            break

    return create_drone_cfg_with_collision_scale(scale)

# In your training loop
def train_step():
    global step_count
    step_count += 1

    if step_count % 100 == 0:  # Update every 100 steps
        new_cfg = get_step_based_cfg(step_count)
        env.update_robot_config(new_cfg)
```

### Method 4: Mixed Curriculum

Combine multiple factors for sophisticated curriculum:

```python
def get_mixed_curriculum_cfg(success_rate, steps, gate_difficulty, crash_frequency):
    """Advanced curriculum considering multiple metrics."""
    from assets.five_in_drone import create_drone_cfg_with_collision_scale

    # Base scale from success rate (0.7 to 1.1 range)
    base_scale = 0.7 + (success_rate * 0.4)

    # Adjust for training progress
    progress_factor = min(steps / 20000, 1.0)
    progress_adjustment = -0.1 * progress_factor  # Gradually make harder

    # Adjust for gate difficulty
    if gate_difficulty == "hard":
        gate_adjustment = 0.1  # Make easier for hard gates
    else:
        gate_adjustment = 0.0

    # Adjust for crash frequency
    if crash_frequency > 0.5:  # Crashing >50% of time
        crash_adjustment = 0.15  # Make much easier
    elif crash_frequency > 0.2:  # Crashing 20-50% of time
        crash_adjustment = 0.05  # Make slightly easier
    else:
        crash_adjustment = 0.0

    # Combine all factors
    final_scale = base_scale + progress_adjustment + gate_adjustment + crash_adjustment

    # Clamp to reasonable bounds
    final_scale = max(0.8, min(1.4, final_scale))

    return create_drone_cfg_with_collision_scale(final_scale)
```

## Implementation Notes

### Updating Environment Configuration

To update the drone configuration during training:

```python
class DroneRacingEnv:
    def __init__(self, robot_cfg=None):
        self.robot_cfg = robot_cfg or FIVE_IN_DRONE
        self.setup_robot()

    def setup_robot(self):
        # Use self.robot_cfg to spawn/configure robot
        self.robot = Articulation(self.robot_cfg)

    def update_robot_config(self, new_cfg):
        """Update robot configuration during training."""
        # Remove old robot
        if hasattr(self, 'robot'):
            self.robot.destroy()

        # Create new robot with updated config
        self.robot_cfg = new_cfg
        self.setup_robot()
```

### Monitoring Progress

Track curriculum effectiveness:

```python
def monitor_curriculum_progress(env, curriculum_history):
    """Monitor and log curriculum learning progress."""
    current_scale = env.robot_cfg.spawn.scale[0]  # Get current scale

    metrics = {
        "current_scale": current_scale,
        "success_rate": calculate_success_rate(),
        "average_crashes": calculate_crash_rate(),
        "gate_completion": calculate_gate_completion()
    }

    curriculum_history.append(metrics)

    # Log progress
    print(f"Step {env.step_count}: Scale={current_scale:.2f}, "
          f"Success={metrics['success_rate']:.2%}, "
          f"Crashes={metrics['average_crashes']:.2f}")

    return metrics
```

### Best Practices

1. **Start Easy**: Begin with larger collision areas (1.2-1.3x scale)
2. **Gradual Progression**: Reduce collision area slowly over time
3. **Performance-Based**: Monitor success rates and adjust accordingly
4. **Plateau Detection**: If performance stagnates, consider making task easier
5. **Safety Margins**: Don't make collision areas too small (<0.8x) to avoid impossible scenarios
6. **Consistent Evaluation**: Test on normal collision area periodically to gauge true performance

## Example Training Script

```python
#!/usr/bin/env python
"""Complete example of curriculum learning for drone racing."""

from assets.five_in_drone import create_drone_cfg_with_collision_scale
from your_env import DroneRacingEnv

def curriculum_learning_train():
    env = DroneRacingEnv()
    step_count = 0
    episode_count = 0
    curriculum_history = []

    # Curriculum parameters
    initial_scale = 1.3
    final_scale = 0.9
    curriculum_duration = 30000  # steps

    while step_count < total_training_steps:
        # Calculate current curriculum scale
        progress = min(step_count / curriculum_duration, 1.0)
        current_scale = initial_scale - (progress * (initial_scale - final_scale))

        # Update environment every 1000 steps
        if step_count % 1000 == 0:
            new_cfg = create_drone_cfg_with_collision_scale(current_scale)
            env.update_robot_config(new_cfg)

            print(f"Updated collision scale to {current_scale:.2f}")

        # Run training steps
        for _ in range(100):
            obs = env.reset()
            done = False

            while not done:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                step_count += 1

                # Track performance
                if done:
                    episode_count += 1
                    curriculum_history.append({
                        'step': step_count,
                        'scale': current_scale,
                        'success': info.get('success', False),
                        'crash': info.get('crash', False)
                    })

    return curriculum_history

if __name__ == "__main__":
    history = curriculum_learning_train()
    print("Training completed!")
```

This curriculum learning approach helps the drone learn progressively, starting with easier collision detection and gradually increasing the precision required for successful navigation.
# Hydra Configuration Usage Analysis

## Overview
This project uses Hydra configuration, but only in a limited capacity.

## Where Hydra is Used

### Training Script (`scripts/rl/train.py`)
- **Import**: `from isaaclab_tasks.utils.hydra import hydra_task_config` (line 93)
- **Decorator**: `@hydra_task_config(args_cli.task, agent_cfg_entry_point)` (line 102)
- **Argument Handling**:
  - `args_cli, hydra_args = parser.parse_known_args()` (line 45)
  - `sys.argv = [sys.argv[0]] + hydra_args` (line 51) - clearing sys.argv for Hydra

### Play Script (`scripts/rl/play.py`)
- **Does NOT use Hydra**
- Uses regular argument parsing and configuration loading via `load_cfg_from_registry`

## Purpose of Hydra Usage

Hydra is used in the training script to:
1. Load task and agent configurations from the IsaacLab framework's configuration registry
2. Handle configuration overrides from command line arguments
3. Manage the experiment configuration structure

## Configuration Files
- Agent configuration: `tasks/drone_racer/agents/skrl_cfg.yaml`
- Environment configurations are loaded from IsaacLab framework registry
- Log configurations are generated automatically during training runs

## Summary
- **Training**: Uses Hydra for configuration management
- **Inference/Playing**: Does not use Hydra
- **Framework**: IsaacLab provides the `hydra_task_config` utility that wraps Hydra functionality
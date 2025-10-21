#!/usr/bin/env python3
"""
Reward Design Basics for RL Beginners
=====================================

This tutorial visualizes and explains common reward functions used in reinforcement learning,
particularly for navigation tasks like drone racing. Understanding reward design is crucial
because rewards directly shape agent behavior during training.

Author: Generated for Drone Racer Project
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Callable, List, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

_SCRIPT_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _SCRIPT_DIR / "figures"
_FIGURES_DIR.mkdir(exist_ok=True)

def _maybe_show(fig):
    """Show figures only when an interactive backend is available."""
    backend = plt.get_backend().lower()
    if 'agg' in backend or os.environ.get("DISPLAY", "") == "":
        plt.close(fig)
    else:
        plt.show()

def _save_fig(fig: plt.Figure, filename: str, **kwargs):
    """Save figures into the tutorial's local figures directory."""
    output_path = _FIGURES_DIR / filename
    fig.savefig(output_path, **kwargs)

class RewardFunction:
    """Base class for reward functions with plotting capabilities."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError

    def plot(self, distances: np.ndarray, ax=None, **kwargs):
        """Plot the reward function over given distance range."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        rewards = self(distances)
        label = kwargs.pop('label', self.name)
        linewidth = kwargs.pop('linewidth', 3)
        ax.plot(distances, rewards, linewidth=linewidth, label=label, **kwargs)
        ax.set_xlabel('Distance to Target (m)', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'{self.name}\n{self.description}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

# ============================================================================
# TYPICAL / CLASSICAL REWARD FUNCTIONS
# ============================================================================

class LinearReward(RewardFunction):
    """Simple linear reward: decreases linearly with distance."""

    def __init__(self):
        super().__init__(
            "Linear Reward",
            "Reward decreases linearly with distance. Simple but can have issues at large distances."
        )
        self.max_distance = 10.0
        self.max_reward = 1.0

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        distance = np.clip(distance, 0, self.max_distance)
        return self.max_reward * (1 - distance / self.max_distance)

class QuadraticReward(RewardFunction):
    """Quadratic (L2) reward: penalizes squared distance."""

    def __init__(self):
        super().__init__(
            "Quadratic (L2) Reward",
            "Quadratic penalty on distance. Common in optimization but can be harsh."
        )
        self.max_distance = 10.0
        self.scale = 0.01

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return -self.scale * distance**2

class BinaryReward(RewardFunction):
    """Binary reward: 1 if at target, 0 otherwise."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(
            "Binary Reward",
            f"Binary reward: 1 if distance < {threshold}m, 0 otherwise. Very sparse but clear."
        )
        self.threshold = threshold

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return (distance < self.threshold).astype(float)

class StepReward(RewardFunction):
    """Step function with multiple thresholds."""

    def __init__(self):
        super().__init__(
            "Step Reward",
            "Discrete rewards at different distance thresholds. More informative than binary."
        )
        self.thresholds = [1.0, 2.0, 5.0, 10.0]
        self.rewards = [1.0, 0.75, 0.5, 0.25, 0.0]

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        distance = np.asarray(distance)
        rewards = np.zeros_like(distance, dtype=float)

        for i, threshold in enumerate(self.thresholds):
            rewards[distance < threshold] = self.rewards[i]

        return rewards

# ============================================================================
# MODERN / ADVANCED REWARD FUNCTIONS
# ============================================================================

class ExponentialReward(RewardFunction):
    """Exponential decay reward (your suggestion)."""

    def __init__(self, std: float = 2.0):
        super().__init__(
            f"Exponential Decay (σ={std})",
            "Smooth exponential decay: exp(-distance/σ). Popular for navigation tasks."
        )
        self.std = std

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return np.exp(-distance / self.std)

class GaussianReward(RewardFunction):
    """Gaussian-shaped reward."""

    def __init__(self, std: float = 2.0):
        super().__init__(
            f"Gaussian Reward (σ={std})",
            "Gaussian-shaped reward: exp(-(distance/σ)²). Very smooth and differentiable."
        )
        self.std = std

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return np.exp(-(distance / self.std)**2)

class TanhReward(RewardFunction):
    """Tanh-based reward function."""

    def __init__(self, std: float = 2.0):
        super().__init__(
            f"Tanh Reward (σ={std})",
            "Tanh normalization: 1 - tanh(distance/σ). Bounded and smooth."
        )
        self.std = std

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return 1 - np.tanh(distance / self.std)

class InverseReward(RewardFunction):
    """Inverse distance reward."""

    def __init__(self, scale: float = 1.0, epsilon: float = 0.1):
        super().__init__(
            f"Inverse Reward (ε={epsilon})",
            "Inverse distance: scale/(distance + ε). High reward when close, bounded away from infinity."
        )
        self.scale = scale
        self.epsilon = epsilon

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return self.scale / (distance + self.epsilon)

class SigmoidReward(RewardFunction):
    """Sigmoid-shaped reward function."""

    def __init__(self, midpoint: float = 5.0, steepness: float = 1.0):
        super().__init__(
            f"Sigmoid Reward (mid={midpoint}, k={steepness})",
            "Sigmoid function: 1/(1 + exp(k*(distance-mid))). S-shaped transition."
        )
        self.midpoint = midpoint
        self.steepness = steepness

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(self.steepness * (distance - self.midpoint)))

class PowerLawReward(RewardFunction):
    """Power law decay reward."""

    def __init__(self, exponent: float = 2.0, scale: float = 1.0):
        super().__init__(
            f"Power Law (α={exponent})",
            "Power law decay: scale/(1 + distance^α). Flexible shape controlled by exponent."
        )
        self.exponent = exponent
        self.scale = scale

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        return self.scale / (1 + distance**self.exponent)

# ============================================================================
# HYBRID AND COMPOSITE REWARDS
# ============================================================================

class HybridReward(RewardFunction):
    """Combination of multiple reward functions."""

    def __init__(self):
        super().__init__(
            "Hybrid Reward",
            "Combination: Exponential + small bonus for very close distances. Best of both worlds."
        )
        self.exp_reward = ExponentialReward(std=3.0)
        self.close_bonus_threshold = 1.0
        self.close_bonus = 0.5

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        exp_reward = self.exp_reward(distance)
        close_bonus = self.close_bonus * (distance < self.close_bonus_threshold).astype(float)
        return exp_reward + close_bonus

class ShapedReward(RewardFunction):
    """Reward shaping with progress bonuses."""

    def __init__(self, std: float = 2.0, previous_distance: np.ndarray | float = None):
        super().__init__(
            "Progress-Shaped Reward",
            "Base exponential reward + progress bonus. Rewards getting closer over time."
        )
        self.exp_reward = ExponentialReward(std)
        self.prev_distance = previous_distance
        self.progress_scale = 0.1

    def __call__(self, distance: np.ndarray | float) -> np.ndarray | float:
        base_reward = self.exp_reward(distance)

        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            progress_bonus = self.progress_scale * np.maximum(progress, 0)
            base_reward += progress_bonus

        return base_reward

def create_comparison_plot():
    """Create comprehensive comparison of all reward functions."""
    distances = np.linspace(0, 10, 500)

    # Create subplots
    fig = plt.figure(figsize=(20, 12))

    # Classical rewards
    ax1 = plt.subplot(2, 3, 1)
    classical_rewards = [
        LinearReward(),
        QuadraticReward(),
        BinaryReward(),
        StepReward()
    ]

    for reward in classical_rewards:
        reward.plot(distances, ax=ax1)
    ax1.set_title('Classical Reward Functions', fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Modern smooth rewards
    ax2 = plt.subplot(2, 3, 2)
    modern_rewards = [
        ExponentialReward(std=2.0),
        GaussianReward(std=2.0),
        TanhReward(std=2.0),
        InverseReward(scale=2.0)
    ]

    for reward in modern_rewards:
        reward.plot(distances, ax=ax2)
    ax2.set_title('Modern Smooth Reward Functions', fontsize=16, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Parameter sensitivity
    ax3 = plt.subplot(2, 3, 3)
    stds = [1.0, 2.0, 4.0]
    for std in stds:
        reward = ExponentialReward(std=std)
        reward.plot(distances, ax=ax3, label=f'σ={std}')
    ax3.set_title('Exponential Reward: Parameter Sensitivity', fontsize=16, fontweight='bold')
    ax3.legend()

    # Hybrid and advanced
    ax4 = plt.subplot(2, 3, 4)
    advanced_rewards = [
        HybridReward(),
        SigmoidReward(midpoint=5.0, steepness=1.0),
        PowerLawReward(exponent=2.0)
    ]

    for reward in advanced_rewards:
        reward.plot(distances, ax=ax4)
    ax4.set_title('Hybrid & Advanced Rewards', fontsize=16, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Reward gradients (important for learning)
    ax5 = plt.subplot(2, 3, 5)
    for reward in [ExponentialReward(std=2.0), GaussianReward(std=2.0), TanhReward(std=2.0)]:
        rewards = reward(distances)
        gradients = np.gradient(rewards, distances)
        ax5.plot(distances, gradients, linewidth=3, label=reward.name)
    ax5.set_xlabel('Distance to Target (m)', fontsize=12)
    ax5.set_ylabel('Reward Gradient', fontsize=12)
    ax5.set_title('Reward Gradients (Learning Signal)', fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Practical comparison table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    comparison_data = [
        ['Function', 'Range', 'Smoothness', 'Gradient', 'Best Use Case'],
        ['Linear', '[0, 1]', 'Low', 'Constant', 'Simple tasks'],
        ['Quadratic', '(-∞, 0]', 'Medium', 'Linear', 'Optimization'],
        ['Binary', '{0, 1}', 'None', 'Zero', 'Sparse rewards'],
        ['Exponential', '(0, 1]', 'High', 'Exponential', 'Navigation'],
        ['Gaussian', '(0, 1]', 'Very High', 'Gaussian', 'Precision tasks'],
        ['Tanh', '[0, 1]', 'High', 'Sigmoid', 'Bounded rewards'],
        ['Hybrid', '[0, 1.5]', 'High', 'Variable', 'Complex tasks']
    ]

    table = ax6.table(cellText=comparison_data[1:],
                     colLabels=comparison_data[0],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('Reward Function Comparison', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    _save_fig(fig, 'reward_functions_comparison.png', dpi=300, bbox_inches='tight')
    _maybe_show(fig)

def create_practical_examples():
    """Create practical examples for drone navigation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    distances = np.linspace(0, 15, 500)

    # Example 1: Gate approaching (exponential is great)
    ax1 = axes[0, 0]
    gate_reward = ExponentialReward(std=2.0)
    gate_reward.plot(distances, ax=ax1, color='green', linewidth=3)
    ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='Gate boundary')
    ax1.fill_between([0, 1.5], [0, 0], [1.2, 1.2], alpha=0.2, color='green', label='Success zone')
    ax1.set_title('Example 1: Gate Approaching Task\n(Exponential Reward)', fontsize=14, fontweight='bold')
    ax1.legend()

    # Example 2: Precision landing (Gaussian is better)
    ax2 = axes[0, 1]
    landing_reward = GaussianReward(std=0.5)
    landing_reward.plot(distances, ax=ax2, color='blue', linewidth=3)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Landing pad radius')
    ax2.fill_between([0, 0.5], [0, 0], [1.2, 1.2], alpha=0.2, color='blue', label='Success zone')
    ax2.set_title('Example 2: Precision Landing\n(Gaussian Reward, σ=0.5)', fontsize=14, fontweight='bold')
    ax2.legend()

    # Example 3: Following a path (Tanh for bounded behavior)
    ax3 = axes[1, 0]
    path_reward = TanhReward(std=3.0)
    path_reward.plot(distances, ax=ax3, color='orange', linewidth=3)
    ax3.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Acceptable deviation')
    ax3.fill_between([0, 2.0], [0, 0], [1.2, 1.2], alpha=0.2, color='orange', label='Good tracking')
    ax3.set_title('Example 3: Path Following\n(Tanh Reward, σ=3.0)', fontsize=14, fontweight='bold')
    ax3.legend()

    # Example 4: Hybrid approach for complex tasks
    ax4 = axes[1, 1]
    hybrid_reward = HybridReward()
    hybrid_reward.plot(distances, ax=ax4, color='purple', linewidth=3)
    ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Close bonus threshold')
    ax4.axvline(x=5.0, color='green', linestyle='--', alpha=0.7, label='Good distance')
    ax4.set_title('Example 4: Complex Navigation\n(Hybrid Reward)', fontsize=14, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    _save_fig(fig, 'practical_reward_examples.png', dpi=300, bbox_inches='tight')
    _maybe_show(fig)

def print_recommendations():
    """Print practical recommendations for reward design."""
    recommendations = """

🎯 REWARD DESIGN RECOMMENDATIONS FOR DRONE NAVIGATION
===================================================

1. 📊 START SIMPLE
   - Begin with exponential decay: reward = exp(-distance/σ)
   - Good default: σ = 2.0 for your single gate task
   - Easy to understand and tune

2. 🔧 TUNING PARAMETERS
   - σ (std) controls how quickly reward decays
   - Larger σ = more forgiving, gentler learning curve
   - Smaller σ = more precise, harder but potentially better performance

3. ⚖️ BALANCE REWARDS
   - Keep distance rewards in range [0, 1]
   - Add sparse bonuses for achievements (gate passed: +100)
   - Avoid reward magnitudes that differ by >1000x

4. 📈 CHECK GRADIENTS
   - Reward should provide useful learning signal
   - Avoid flat regions (gradient = 0) except at completion
   - Smooth functions train better than discontinuous ones

5. 🧪 TEST VISUALLY
   - Always plot your reward function first
   - Check that it makes sense for your task
   - Ensure rewards encourage desired behavior

6. 🎪 COMBINE WISELY
   - Distance reward + achievement bonus = good combination
   - Consider progress shaping: reward getting closer over time
   - Multiple terms should work together, not compete

FOR YOUR SINGLE GATE TASK:
--------------------------
✅ Recommended: reward = exp(-distance/2.0)
✅ Add bonus: +100 for successful gate pass
✅ Penalty: -10 for going out of bounds
❌ Avoid: Complex coordinated flight penalties initially

REMEMBER: Good reward design is an iterative process!
Start simple, test, then add complexity only if needed.

    """
    print(recommendations)

if __name__ == "__main__":
    print("🚁 Reward Design Tutorial for Drone Racing Beginners 🚁")
    print("=" * 60)

    # Create comprehensive comparison
    print("\n📊 Generating comprehensive reward function comparison...")
    create_comparison_plot()

    # Create practical examples
    print("\n🎯 Generating practical drone navigation examples...")
    create_practical_examples()

    # Print recommendations
    print_recommendations()

    print(f"\n✅ Tutorial complete!")
    print(f"📁 Plots saved to:")
    print(f"   - tutorials/reward_functions_comparison.png")
    print(f"   - tutorials/practical_reward_examples.png")
    print(f"\n💡 Key takeaway: For your single gate task, start with:")
    print(f"   reward = torch.exp(-distance / 2.0)")

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def pos_error_l2(current_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """
    Calculates the squared L2 distance penalty.
    This is a PENALTY function.
    """
    return torch.sum(torch.square(current_pos - target_pos), dim=1)


def pos_error_tanh(current_pos: torch.Tensor, target_pos: torch.Tensor, std: float) -> torch.Tensor:
    """
    Calculates the tanh-based distance reward.
    This is a REWARD function.
    """
    distance = torch.norm(current_pos - target_pos, dim=1)
    return 1.0 - torch.tanh(distance / std)


# --- Simulation Setup ---
# Define a target at the origin
target = torch.tensor([[0.0]], dtype=torch.float32)

# Create a range of distances from the target to test
distances = np.linspace(0, 10, 200)
# Create corresponding positions (we only need one dimension for this example)
positions = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)

# --- Calculate Rewards/Penalties ---
# Calculate the L2 penalty
l2_penalty = pos_error_l2(positions, target)
# We take the negative to visualize it as a "reward shape" (higher is better)
l2_as_reward = -l2_penalty

# Calculate the tanh reward with two different `std` values to see its effect
tanh_reward_std1 = pos_error_tanh(positions, target, std=1.0)
tanh_reward_std3 = pos_error_tanh(positions, target, std=3.0)


# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(distances, l2_as_reward.numpy(), label='-pos_error_l2 (Negative Penalty)', color='red', linestyle='--')
ax.plot(distances, tanh_reward_std1.numpy(), label='pos_error_tanh (std=1.0)', color='blue', linewidth=2)
ax.plot(distances, tanh_reward_std3.numpy(), label='pos_error_tanh (std=3.0)', color='green', linewidth=2)

ax.set_title('Comparison of Reward Function Shapes', fontsize=16)
ax.set_xlabel('Distance from Target', fontsize=12)
ax.set_ylabel('Reward Value', fontsize=12)
ax.legend(fontsize=10)
ax.axhline(0, color='black', linewidth=0.5)  # Add a line at y=0
ax.set_ylim(-10, 1.2)  # Set y-limits to better see the tanh curve

current_file_folder = os.path.dirname(os.path.abspath(__file__))
saved_folder = os.path.join(current_file_folder, 'figures')
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

plt.savefig(os.path.join(saved_folder, 'reward_function_comparison.png'))
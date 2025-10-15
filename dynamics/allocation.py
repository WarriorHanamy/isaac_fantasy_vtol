# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class Allocation:
    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        base_matrix = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = base_matrix.unsqueeze(0).repeat(num_envs, 1, 1)
        self._allocation_matrix_inv = torch.linalg.pinv(base_matrix).unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    @property
    def allocation_matrix(self) -> torch.Tensor:
        return self._allocation_matrix

    @property
    def allocation_matrix_inv(self) -> torch.Tensor:
        return self._allocation_matrix_inv

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque

    def motor_thrust_from_wrench(self, wrench: torch.Tensor) -> torch.Tensor:
        """Compute individual motor thrusts from desired body wrench.

        Args:
            wrench: Tensor (num_envs, 4) => [total_thrust, mx, my, mz]

        Returns:
            motor_thrusts: Tensor (num_envs, 4) with per-rotor thrust.
        """
        return torch.bmm(self._allocation_matrix_inv, wrench.unsqueeze(-1)).squeeze(-1)

    def omega_from_wrench(self, wrench: torch.Tensor, clamp: tuple[float, float] | None = None) -> torch.Tensor:
        """Compute rotor angular velocities required to realize a target body wrench."""
        motor_thrusts = self.motor_thrust_from_wrench(wrench)
        if clamp is not None:
            motor_thrusts = motor_thrusts.clamp(clamp[0], clamp[1])
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)
        omega = torch.sqrt(torch.clamp(motor_thrusts / self._thrust_coeff, min=0.0))
        return omega

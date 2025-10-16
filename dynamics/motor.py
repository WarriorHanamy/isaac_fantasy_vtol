# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from collections.abc import Sequence

import torch


class Motor:
    def __init__(
        self,
        num_envs: int,
        taus: Sequence[float] | torch.Tensor,
        init: Sequence[float] | torch.Tensor,
        max_rate: Sequence[float] | torch.Tensor,
        min_rate: Sequence[float] | torch.Tensor,
        dt: float,
        use: bool,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - taus: (4,) Tensor or list specifying time constants per motor.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - max_rate: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rate: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        self.num_envs: int = num_envs
        self.num_motors: int = len(taus)
        self.dt: float = dt
        self.use: bool = use
        self.init: Sequence[float] | torch.Tensor = init
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype

        self.omega: torch.Tensor = (
            torch.tensor(init, device=device, dtype=dtype).expand(num_envs, -1).clone()
        )  # (num_envs, num_motors)

        # Convert to tensors and expand for all drones
        self.tau: torch.Tensor = torch.tensor(taus, device=device, dtype=dtype).expand(num_envs, -1)
        self.max_rate: torch.Tensor = torch.tensor(max_rate, device=device, dtype=dtype).expand(num_envs, -1)
        self.min_rate: torch.Tensor = torch.tensor(min_rate, device=device, dtype=dtype).expand(num_envs, -1)

    def compute(self, omega_ref: torch.Tensor) -> torch.Tensor:
        """
        Computes the new omega values based on reference omega and motor dynamics.

        Parameters:
        - omega_ref: (num_envs, num_motors) Tensor of reference omega values.

        Returns:
        - omega: (num_envs, num_motors) Tensor of updated omega values.
        """

        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Compute omega rate using first-order motor dynamics
        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)  # (num_envs, num_motors)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        # Integrate
        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """
        Resets the motor model to initial conditions.
        """
        count = len(env_ids)
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(count, -1)

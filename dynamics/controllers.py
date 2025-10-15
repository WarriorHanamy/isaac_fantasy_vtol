from __future__ import annotations

from typing import Sequence

import torch


class BodyRateController:
    """Simple body-rate feedback controller inspired by Lee-rate control.

    The controller computes body torques that drive the measured angular rates
    towards commanded rates while adding a gyroscopic feedforward term
    ``omega x (I * omega)``.
    """

    def __init__(
        self,
        num_envs: int,
        rate_gains: Sequence[float],
        inertia: Sequence[float],
        max_body_rates: Sequence[float],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if len(rate_gains) != 3 or len(inertia) != 3 or len(max_body_rates) != 3:
            raise ValueError("rate_gains, inertia and max_body_rates must have length 3.")

        gains_tensor = torch.tensor(rate_gains, dtype=dtype, device=device).view(1, 3)
        inertia_tensor = torch.tensor(inertia, dtype=dtype, device=device).view(1, 3)
        max_rates_tensor = torch.tensor(max_body_rates, dtype=dtype, device=device).view(1, 3)

        self._rate_gains = gains_tensor.expand(num_envs, -1)
        self._inertia = inertia_tensor.expand(num_envs, -1)
        self._max_body_rates = max_rates_tensor.expand(num_envs, -1)

    def compute(self, current_rates: torch.Tensor, desired_rates: torch.Tensor) -> torch.Tensor:
        """Return body torque command for the given desired angular rates."""
        if current_rates.shape != desired_rates.shape:
            raise ValueError("current_rates and desired_rates must have the same shape.")

        desired_clamped = torch.clamp(desired_rates, -self._max_body_rates, self._max_body_rates)
        rate_error = current_rates - desired_clamped
        feedforward = torch.cross(current_rates, self._inertia * current_rates, dim=1)
        torque = -self._rate_gains * rate_error + feedforward
        return torque

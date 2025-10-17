# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import Allocation, BodyRateController, Motor
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._control_mode = self.cfg.control_mode.lower()
        max_motor_thrust = self.cfg.thrust_coef * (self.cfg.omega_max**2)
        self._max_motor_thrust = torch.tensor(max_motor_thrust, dtype=self._raw_actions.dtype, device=self.device)
        self._max_motor_thrust_value = float(max_motor_thrust)
        self._total_thrust_max = self._max_motor_thrust * 4.0
        self._max_body_rate = torch.tensor(
            self.cfg.max_body_rate, dtype=self._raw_actions.dtype, device=self.device
        ).view(1, 3)
        self._body_rate_controller: BodyRateController | None = None
        if self._control_mode == "body_rate":
            self._body_rate_controller = BodyRateController(
                num_envs=self.num_envs,
                rate_gains=self.cfg.rate_gains,
                inertia=self.cfg.body_inertia,
                max_body_rates=self.cfg.max_body_rate,
                device=self.device,
                dtype=self._raw_actions.dtype,
            )
        elif self._control_mode != "motor":
            raise ValueError(f"Unsupported control_mode '{self.cfg.control_mode}'.")
        self._throttle_min = float(self.cfg.throttle_limits[0])
        self._throttle_max = float(self.cfg.throttle_limits[1])

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions
        clamped = self._raw_actions.clamp_(-1.0, 1.0)

        if self._control_mode == "motor":
            mapped = (clamped + 1.0) / 2.0
            omega_ref = self.cfg.omega_max * mapped
        else:
            throttle_cmd = clamped[:, 0]
            throttle_frac = (throttle_cmd + 1.0) * 0.5
            throttle_frac = throttle_frac.clamp(0.0, 1.0)
            throttle_frac = self._throttle_min + throttle_frac * (self._throttle_max - self._throttle_min)
            total_thrust = throttle_frac * self._total_thrust_max

            desired_rates = clamped[:, 1:] * self._max_body_rate
            current_rates = self._robot.data.root_ang_vel_b
            if self._body_rate_controller is None:
                raise RuntimeError("Body-rate controller not initialised.")
            torque_cmd = self._body_rate_controller.compute(current_rates, desired_rates)
            wrench_cmd = torch.cat([total_thrust.unsqueeze(1), torque_cmd], dim=1)
            omega_ref = self._allocation.omega_from_wrench(wrench_cmd, clamp=(0.0, self._max_motor_thrust_value))

            log(self._env, ["throttle_cmd"], total_thrust.unsqueeze(1))
            log(self._env, ["rate_cmd_x", "rate_cmd_y", "rate_cmd_z"], desired_rates)
            log(self._env, ["torque_cmd_x", "torque_cmd_y", "torque_cmd_z"], torque_cmd)

        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)

        log(self._env, ["a1", "a2", "a3", "a4"], self._raw_actions)
        log(self._env, ["w1", "w2", "w3", "w4"], omega_real)

    def apply_actions(self):
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._motor.reset(env_ids)
        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    arm_length: float = 0.035
    """Length of the arms of the drone in meters."""
    drag_coef: float = 1.5e-9
    """Drag torque coefficient."""
    thrust_coef: float = 2.25e-7
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
    thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
    omega_max: float = 5145.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
    1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
    taus: list[float] = [0.0001, 0.0001, 0.0001, 0.0001]
    """Time constants for each motor."""
    init: list[float] = [2572.5, 2572.5, 2572.5, 2572.5]
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = [50000.0, 50000.0, 50000.0, 50000.0]    
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = [-50000.0, -50000.0, -50000.0, -50000.0]
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""
    control_mode: str = "body_rate"
    """Control strategy: 'motor' for direct rotor commands, 'body_rate' for throttle + rate commands."""
    rate_gains: tuple[float, float, float] = (0.02, 0.02, 0.01)
    """Proportional gains used by the body-rate controller."""
    max_body_rate: tuple[float, float, float] = (10.0, 10.0, 5.0)
    """Maximum absolute commanded body rates in rad/s."""
    throttle_limits: tuple[float, float] = (0.0, 1.0)
    """Throttle fraction limits applied after action mapping."""
    body_inertia: tuple[float, float, float] = (0.003, 0.003, 0.006)
    """Diagonal inertia values (kg·m²) used in the body-rate controller."""

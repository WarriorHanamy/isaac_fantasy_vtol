import pytest
import torch

from dynamics import Allocation, BodyRateController


@pytest.fixture
def allocation():
    num_envs = 2
    arm_length = 0.1
    thrust_coeff = 1e-6
    drag_coeff = 1e-7
    return Allocation(
        num_envs=num_envs,
        arm_length=arm_length,
        thrust_coeff=thrust_coeff,
        drag_coeff=drag_coeff,
    ), num_envs, thrust_coeff


def test_allocation_compute_shape(allocation):
    alloc, num_envs, _ = allocation
    omega = torch.full((num_envs, 4), 2000.0)
    result = alloc.compute(omega)
    assert result.shape == (num_envs, 4)
    assert torch.all(result[:, 0] > 0.0)


def test_allocation_roundtrip(allocation):
    alloc, num_envs, thrust_coeff = allocation
    omega = torch.tensor([[1000.0, 900.0, 1100.0, 800.0]]).repeat(num_envs, 1)
    wrench = alloc.compute(omega)
    omega_recovered = alloc.omega_from_wrench(wrench)
    wrench_recovered = alloc.compute(omega_recovered)
    assert torch.allclose(wrench_recovered, wrench, rtol=1e-4, atol=1e-4)
    thrust_recovered = alloc.motor_thrust_from_wrench(wrench)
    assert torch.all(thrust_recovered >= 0.0)
    wrench_from_thrust = torch.bmm(alloc.allocation_matrix, thrust_recovered.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(wrench_from_thrust, wrench, rtol=1e-4, atol=1e-4)


@pytest.fixture
def body_rate_controller():
    num_envs = 2
    gains = (0.02, 0.02, 0.01)
    inertia = (0.003, 0.003, 0.006)
    max_rates = (10.0, 10.0, 5.0)
    controller = BodyRateController(
        num_envs=num_envs,
        rate_gains=gains,
        inertia=inertia,
        max_body_rates=max_rates,
    )
    return controller, torch.tensor(max_rates)


def test_body_rate_controller_zero_error(body_rate_controller):
    controller, _ = body_rate_controller
    current = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    desired = current.clone()
    torque = controller.compute(current, desired)
    desired_clamped = torch.clamp(desired, -controller._max_body_rates, controller._max_body_rates)  # pylint: disable=protected-access
    rate_error = current - desired_clamped
    expected = -controller._rate_gains * rate_error  # pylint: disable=protected-access
    feedforward = torch.cross(current, controller._inertia * current, dim=1)  # pylint: disable=protected-access
    expected = expected + feedforward
    assert torch.allclose(torque, expected, rtol=1e-5, atol=1e-5)


def test_body_rate_controller_limits(body_rate_controller):
    controller, max_rates = body_rate_controller
    current = torch.zeros((2, 3))
    desired = torch.tensor([[20.0, -20.0, 10.0], [-20.0, 20.0, -10.0]])
    torque = controller.compute(current, desired)
    assert torch.all((desired.abs() >= max_rates).any(dim=1))
    assert torch.all(torch.isfinite(torque))


def test_body_rate_controller_response(body_rate_controller):
    controller, _ = body_rate_controller
    current = torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
    desired = torch.zeros_like(current)
    torque = controller.compute(current, desired)
    desired_clamped = torch.clamp(desired, -controller._max_body_rates, controller._max_body_rates)  # pylint: disable=protected-access
    rate_error = current - desired_clamped
    expected = -controller._rate_gains * rate_error  # pylint: disable=protected-access
    feedforward = torch.cross(current, controller._inertia * current, dim=1)  # pylint: disable=protected-access
    expected = expected + feedforward
    assert torch.allclose(torque, expected, rtol=1e-5, atol=1e-5)

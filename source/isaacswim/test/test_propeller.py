from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.actuators import Propeller, ThrusterArray, Thrusters  # noqa: E402
from isaacswim.actuators.propeller import advance_speed  # noqa: E402
from isaacswim.control import VehicleState  # noqa: E402

RHO = 1000.0
f64 = torch.float64


def t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


@pytest.fixture
def prop():
    return Propeller.t200(16.0)


def test_t200_full_throttle_matches_the_datasheet(prop):
    fwd, rev = prop.max_thrust(RHO)
    assert fwd == pytest.approx(51.4, rel=0.05)
    assert rev == pytest.approx(39.9, rel=0.05)


def test_bollard_is_the_zero_advance_case(prop):
    n = t(np.linspace(-50, 55, 23))
    assert torch.allclose(prop.thrust(n, torch.zeros_like(n), RHO),
                          prop.bollard_thrust(n, RHO))


def test_thrust_vanishes_at_the_zero_thrust_advance_ratio(prop):
    n = t([10.0, 30.0, 55.0])
    u = prop.j0_fwd * n * prop.diameter
    assert torch.allclose(prop.thrust(n, u, RHO), torch.zeros(3, dtype=f64), atol=1e-12)


def test_thrust_falls_with_speed_and_is_finite_at_zero_rpm(prop):
    n = t([40.0] * 4)
    T = prop.thrust(n, t([0.0, 0.5, 1.0, 1.5]), RHO)
    assert bool((T[1:] < T[:-1]).all())
    z = prop.thrust(t([0.0, 0.0]), t([0.0, 2.0]), RHO)
    assert torch.equal(z, torch.zeros(2, dtype=f64))


def test_crash_stop_brakes_harder_than_bollard(prop):
    n = t([-40.0])
    assert float(prop.thrust(n, t([1.0]), RHO)) < float(prop.bollard_thrust(n, RHO))


def test_rps_for_thrust_inverts_bollard_and_clamps(prop):
    T = t([-30.0, -5.0, 0.0, 5.0, 40.0])
    n = prop.rps_for_thrust(T, RHO)
    assert torch.allclose(prop.bollard_thrust(n, RHO), T, atol=1e-9)
    big = prop.rps_for_thrust(t([1e4, -1e4]), RHO)
    assert torch.allclose(big, t([prop.max_rps_fwd, -prop.max_rps_rev]))


def test_a_measured_curve_replaces_the_line(prop):
    J = np.linspace(0.0, 1.2, 13)
    line = Propeller.t200(16.0, kt_curve=(J, prop.kt0_fwd * (1 - J / prop.j0_fwd)))
    n = t([20.0, 40.0, 55.0])
    u = t([0.2, 0.9, 1.4])
    assert torch.allclose(line.thrust(n, u, RHO), prop.thrust(n, u, RHO), atol=1e-9)


def test_advance_speed_includes_rotation_and_current():
    pos = t([[1.0, 0.0, 0.0]])
    push = t([[0.0, 1.0, 0.0]])
    u = advance_speed(t([[0.0, 0.5, 0.0]]), t([[0.0, 0.0, 2.0]]), t([[0.0, 0.1, 0.0]]),
                      pos, push)
    assert float(u) == pytest.approx(0.5 + 2.0 - 0.1)


def state(v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0), current=None, E=1):
    z = torch.zeros(E, 3, dtype=f64)
    return VehicleState(pos_w=z, quat_w=torch.tensor([[1.0, 0, 0, 0]] * E, dtype=f64),
                        lin_vel_b=t([v] * E), ang_vel_b=t([w] * E),
                        joint_pos=torch.zeros(E, 0), joint_vel=torch.zeros(E, 0),
                        current_b=None if current is None else t([current] * E))


def one_thruster(prop, **kw):
    arr = ThrusterArray.from_geometry([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    act = Thrusters(arr, input="thrust", propeller=prop, rho=RHO, **kw)
    act.bind(body_names=["base_link"], joint_names=[], num_envs=1, device="cpu")
    return act


def test_thrusters_deliver_the_command_at_rest_and_less_underway(prop):
    act = one_thruster(prop)
    assert act.needs_state
    assert act.array.fwd == pytest.approx(prop.max_thrust(RHO)[0])
    act.set_setpoint([30.0])
    _, f, _ = act.wrench(0.01, state())
    assert float(f[0, 0]) == pytest.approx(30.0)
    _, f, _ = act.wrench(0.01, state(v=(1.0, 0.0, 0.0)))
    assert 0.0 < float(f[0, 0]) < 30.0
    _, f, _ = act.wrench(0.01, state(v=(1.0, 0.0, 0.0), current=(1.0, 0.0, 0.0)))
    assert float(f[0, 0]) == pytest.approx(30.0)


def test_the_lag_is_on_shaft_speed(prop):
    act = one_thruster(prop, time_constant=0.2)
    act.set_setpoint([30.0])
    _, f, _ = act.wrench(0.2, state())
    n_cmd = float(prop.rps_for_thrust(t([30.0]), RHO))
    assert float(act.rps[0, 0]) == pytest.approx(n_cmd * (1 - np.exp(-1.0)))
    assert float(f[0, 0]) == pytest.approx(30.0 * (1 - np.exp(-1.0)) ** 2)


def test_without_state_the_propeller_assumes_still_water(prop):
    act = one_thruster(prop)
    act.set_setpoint([20.0])
    _, f, _ = act.wrench(0.01, None)
    assert float(f[0, 0]) == pytest.approx(20.0)


from isaacswim.actuators import Motor  # noqa: E402


def motor_thruster(time_constant=0.0, rho=RHO):
    arr = ThrusterArray.from_geometry([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    act = Thrusters(arr, input="thrust", propeller=Propeller.t200(16.0), motor=Motor.t200(16.0),
                    rho=rho, time_constant=time_constant)
    act.bind(body_names=["base_link"], joint_names=[], num_envs=1, device="cpu")
    return act


def settle(act, cmd, steps=400, dt=1 / 120, st=None):
    act.set_setpoint([cmd])
    for _ in range(steps):
        _, f, _ = act.wrench(dt, st or state())
    return float(f[0, 0])


def test_full_duty_speed_matches_blue_robotics():
    act = motor_thruster()
    settle(act, 1e4)
    assert float(act.rps[0, 0]) * 60 == pytest.approx(3533, rel=0.04)
    settle(act, -1e4)
    assert -float(act.rps[0, 0]) * 60 == pytest.approx(3468, rel=0.04)
    assert act.array.fwd == pytest.approx(51.4, rel=0.08)


def test_settles_on_the_commanded_thrust_at_bollard():
    act = motor_thruster()
    for cmd in (5.0, 20.0, -15.0, 0.0):
        assert settle(act, cmd) == pytest.approx(cmd, abs=1e-6)
    assert float(act.current[0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_spin_up_has_the_mechanical_time_constant():
    m = Motor.t200(16.0)
    tau = m.inertia * m.resistance / m.ke ** 2
    act = motor_thruster()
    act.set_setpoint([2.0])
    dt, t, rps = 1e-4, 0.0, []
    while t < 6 * tau:
        act.wrench(dt, state())
        rps.append(float(act.rps[0, 0]))
        t += dt
    rps = np.array(rps)
    t63 = (np.argmax(rps >= 0.632 * rps[-1]) + 1) * dt
    assert 0.5 * tau < t63 < 1.1 * tau
    assert np.all(np.diff(rps) >= -1e-12), "overshoot"


def test_stable_at_steps_much_longer_than_the_rotor():
    act = motor_thruster()
    act.set_setpoint([40.0])
    seen = []
    for _ in range(40):
        act.wrench(0.05, state())
        seen.append(float(act.rps[0, 0]))
    assert np.all(np.diff(seen) >= -1e-9) and np.isfinite(seen).all()
    assert float(act.thrust[0, 0]) == pytest.approx(40.0, rel=1e-3)


def test_underway_the_rotor_unloads_and_thrust_falls():
    act = motor_thruster()
    still = settle(act, 30.0)
    n_still = float(act.rps[0, 0])
    moving = settle(act, 30.0, st=state(v=(1.0, 0.0, 0.0)))
    assert moving < still and float(act.rps[0, 0]) > n_still


def test_supply_current_matches_blue_robotics():
    act = motor_thruster(rho=1000.0)
    settle(act, 30.0)
    i = float(act.current[0, 0])
    assert i == pytest.approx(10.0, rel=0.05)
    assert float(act.power[0]) == pytest.approx(16.0 * i)


def test_motor_needs_torque_coefficients():
    arr = ThrusterArray.from_geometry([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="torque"):
        Thrusters(arr, propeller=Propeller.t200(kq0_fwd=None), motor=Motor.t200())
    with pytest.raises(ValueError, match="torque"):
        Thrusters(arr, motor=Motor.t200())

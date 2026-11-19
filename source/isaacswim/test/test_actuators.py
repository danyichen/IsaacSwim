from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.actuators import JointServos, ThrusterArray, Thrusters  # noqa: E402

POS = [
    [0.14, -0.10, 0.0], [0.14, 0.10, 0.0], [-0.14, -0.10, 0.0], [-0.14, 0.10, 0.0],
    [0.118, -0.215, 0.07], [0.118, 0.215, 0.07], [-0.118, -0.215, 0.07],
    [-0.118, 0.215, 0.07],
]
AX = [
    [-0.866, -0.5, 0.0], [-0.866, 0.5, 0.0], [0.866, -0.5, 0.0], [0.866, 0.5, 0.0],
    [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
]


@pytest.fixture
def array():
    return ThrusterArray.from_geometry(POS, AX)


def bind(act, *, bodies=("base_link", "arm"), joints=(), E=3, body_com=None,
         default_joint_pos=None):
    act.bind(body_names=list(bodies), joint_names=list(joints), num_envs=E,
             device="cpu", body_com=body_com, default_joint_pos=default_joint_pos)
    return act


def test_bluerov_geometry_is_fully_actuated(array):
    assert array.rank == 6
    assert array.null.shape == (8, 2)
    assert array.cond < 20.0


def test_underactuated_array_reports_unreachable_wrench():
    arr = ThrusterArray.from_geometry([[0, 0, 0]], [[1, 0, 0]])
    unreachable = np.array([10.0, 5.0, 0, 0, 0, 0])
    thrust, info = arr.allocate(unreachable)
    assert info["scale"] == 0 and info["saturated"]
    assert np.allclose(arr.A @ thrust, info["scale"] * unreachable)
    act = bind(Thrusters(arr), E=2)
    act.set_setpoint(np.array([unreachable, [10.0, 0, 0, 0, 0, 0]]))
    assert act.info["scale"].tolist() == [0.0, 1.0]
    assert act.info["saturated"].tolist() == [True, False]
    assert act.command[:, 0].tolist() == [0.0, 10.0]


@pytest.mark.parametrize("axis", [[0, 0, 0], [float("nan"), 0, 0]])
def test_invalid_thruster_axis_is_rejected(axis):
    with pytest.raises(ValueError, match="geometry|axes"):
        ThrusterArray.from_geometry([[0, 0, 0]], [axis])


def test_thruster_array_matches_the_original(array):
    pinv, null = array.pinv, array.null
    lo, hi = -array.rev, array.fwd

    def project(w, iters=200):
        base = pinv @ w
        t = base.copy()
        if np.all(t <= hi + 1e-9) and np.all(t >= lo - 1e-9):
            return t, True
        for _ in range(iters):
            t = base + null @ (null.T @ (np.clip(t, lo, hi) - base))
            if np.all(t <= hi + 1e-6) and np.all(t >= lo - 1e-6):
                return np.clip(t, lo, hi), True
        return np.clip(t, lo, hi), False

    def original(w):
        t, ok = project(w)
        if ok:
            return t, 1.0
        lo_s, hi_s, t = 0.0, 1.0, np.zeros(8)
        for _ in range(40):
            mid = 0.5 * (lo_s + hi_s)
            tm, okm = project(w * mid)
            if okm:
                lo_s, t = mid, tm
            else:
                hi_s = mid
        return t, lo_s

    rng = np.random.default_rng(0)
    for scale in (5.0, 60.0, 400.0):
        for _ in range(40):
            w = rng.normal(size=6) * scale
            t_new, info = array.allocate(w)
            t_old, s_old = original(w)
            if not info["saturated"]:
                assert np.array_equal(t_new, t_old)
                assert info["scale"] == s_old
            else:
                assert info["scale"] == pytest.approx(s_old, abs=1e-6)
                assert np.allclose(info["realised"], info["scale"] * w, atol=1e-6)


def test_saturated_scale_is_the_largest_feasible_one(array):
    rng = np.random.default_rng(1)
    for _ in range(40):
        w = rng.normal(size=6) * 400.0
        t, info = array.allocate(w)
        s = info["scale"]
        assert info["saturated"] and 0.0 < s < 1.0
        assert np.all(t <= array.fwd + 1e-9) and np.all(t >= -array.rev - 1e-9)
        assert not array._feasible(w * s * (1.0 + 1e-4))


def test_allocation_degrades_along_the_commanded_direction(array):
    w = np.array([400.0, 0.0, 0.0, 0.0, 0.0, 30.0])
    t, info = array.allocate(w)
    assert info["saturated"] and 0.0 < info["scale"] < 1.0
    assert np.allclose(info["realised"], info["scale"] * w, atol=1e-6)
    assert np.all(t <= array.fwd + 1e-9) and np.all(t >= -array.rev - 1e-9)


def test_rebuild_about_com_moves_only_the_moments(array):
    before = array.A.copy()
    array.rebuild_about_com([0.0031, 0.0002, 0.0076])
    assert np.allclose(array.A[:3], before[:3])
    assert not np.allclose(array.A[3:], before[3:])


def test_json_and_geometry_constructors_agree(tmp_path, array):
    import json

    path = tmp_path / "thrusters.json"
    path.write_text(json.dumps({"frame": "base_link", "thrusters": [
        {"link": f"t{i}", "position": p, "axis": a}
        for i, (p, a) in enumerate(zip(POS, AX))]}))
    assert np.array_equal(ThrusterArray(path).A, array.A)


def test_wrench_setpoint_realises_the_allocators_wrench(array):
    act = bind(Thrusters(array))
    w = np.array([10.0, -5.0, 20.0, 0.5, -1.0, 2.0])
    act.set_setpoint(w)
    body, f, t = act.wrench(1 / 120)
    t_ref, _ = array.allocate(w)
    f_ref, m_ref = array.body_wrench(t_ref)
    assert body == 0
    for e in range(3):
        assert np.allclose(f[e].numpy(), f_ref, atol=1e-12)
        assert np.allclose(t[e].numpy(), m_ref, atol=1e-12)


def test_binding_rebuilds_about_the_body_com(array):
    com = torch.zeros(2, 3)
    com[0] = torch.tensor([0.0031, 0.0002, 0.0076])
    bind(Thrusters(array), body_com=com)
    assert np.allclose(array.com, com[0].numpy())


def test_thrust_setpoint_is_clamped_to_the_asymmetric_limits(array):
    act = bind(Thrusters(array, input="thrust"))
    act.set_setpoint([1e3] * 4 + [-1e3] * 4)
    act.wrench(0.01)
    assert torch.allclose(act.thrust[0, :4], torch.full((4,), array.fwd, dtype=torch.float64))
    assert torch.allclose(act.thrust[0, 4:], torch.full((4,), -array.rev, dtype=torch.float64))


def test_pinv_allocation_is_batched_and_reports_saturation(array):
    act = bind(Thrusters(array, allocation="pinv"), E=2)
    act.set_setpoint(torch.tensor([[1.0, 0, 0, 0, 0, 0], [500.0, 0, 0, 0, 0, 0]]))
    assert act.info["saturated"].tolist() == [False, True]
    _, f, _ = act.wrench(0.01)
    assert f[0, 0].item() == pytest.approx(1.0, abs=1e-9)


def test_time_constant_lags_like_a_first_order_system(array):
    act = bind(Thrusters(array, input="thrust", time_constant=0.2), E=1)
    act.set_setpoint([10.0] * 8)
    dt, n = 0.01, 20
    for _ in range(n):
        act.wrench(dt)
    assert act.thrust[0, 0].item() == pytest.approx(10.0 * (1 - math.exp(-1.0)), rel=1e-6)


def test_setpoint_shape_is_checked(array):
    act = bind(Thrusters(array))
    with pytest.raises(ValueError, match="expected"):
        act.set_setpoint([1.0, 2.0, 3.0])


def test_unbound_actuator_says_what_to_do(array):
    with pytest.raises(RuntimeError, match="not bound"):
        Thrusters(array).set_setpoint(np.zeros(6))


def test_unknown_body_is_rejected(array):
    with pytest.raises(ValueError, match="not in the articulation"):
        bind(Thrusters(array, body="hull"))


JOINTS = ["alpha_axis_e", "alpha_axis_d", "jaw_a", "camera_mount_joint"]


def test_servo_ramp_matches_reference_ramp():
    rate, dt = 0.5, 1.0 / 120.0
    q0 = torch.tensor([[2.85, 2.45, 0.2, 0.0]])
    act = bind(JointServos(["alpha_axis_.*"], rate=rate), joints=JOINTS, E=1,
               default_joint_pos=q0)
    want = {"alpha_axis_e": 3.1, "alpha_axis_d": 2.30}

    ref = {n: np.float32(q0[0, i].item()) for i, n in enumerate(JOINTS[:2])}
    for _ in range(80):
        for name, target in want.items():
            have, step = float(ref[name]), rate * dt
            ref[name] = np.float32(target if abs(target - have) <= step
                                   else have + math.copysign(step, target - have))
        act.set_setpoint(want)
        _, q = act.joint_targets(dt)
        assert q[0, 0].item() == float(ref["alpha_axis_e"])
        assert q[0, 1].item() == float(ref["alpha_axis_d"])
    assert bool(act.at_setpoint[0])


def test_servo_group_order_follows_the_patterns():
    act = bind(JointServos(["camera_mount_joint", "alpha_axis_.*"]), joints=JOINTS)
    assert act.joint_names == ["camera_mount_joint", "alpha_axis_e", "alpha_axis_d"]
    assert act.joint_indices.tolist() == [3, 0, 1]


def test_servo_without_a_rate_limit_steps_immediately():
    act = bind(JointServos(["jaw_a"]), joints=JOINTS, E=2)
    act.set_setpoint({"jaw_a": 0.7})
    _, q = act.joint_targets(0.01)
    assert torch.allclose(q, torch.full((2, 1), 0.7))


def test_servo_rejects_unknown_joints():
    with pytest.raises(ValueError, match="matches no joint"):
        bind(JointServos(["elbow"]), joints=JOINTS)
    act = bind(JointServos(["jaw_a"]), joints=JOINTS)
    with pytest.raises(KeyError):
        act.set_setpoint({"alpha_axis_e": 1.0})


def test_servo_reset_returns_to_the_spawn_pose():
    q0 = torch.tensor([[1.0, 2.0, 0.3, 0.0]] * 2)
    act = bind(JointServos(["alpha_axis_.*"], rate=0.5), joints=JOINTS, E=2,
               default_joint_pos=q0)
    act.set_setpoint([0.0, 0.0])
    for _ in range(50):
        act.joint_targets(0.1)
    act.reset([1])
    assert torch.allclose(act.target[1], torch.tensor([1.0, 2.0]))
    assert not torch.allclose(act.target[0], torch.tensor([1.0, 2.0]))


def test_batched_nullspace_allocator_matches_the_numpy_one(array):
    rng = np.random.default_rng(3)
    W = np.concatenate([rng.normal(scale=s, size=(40, 6)) for s in (10.0, 60.0, 250.0)])
    act = bind(Thrusters(array, input="wrench", allocation="nullspace"), E=len(W))
    act.set_setpoint(torch.as_tensor(W))
    for e, w in enumerate(W):
        t, info = array.allocate(w)
        assert np.allclose(act.command[e].numpy(), t, atol=1e-9), e
        assert float(act.info["scale"][e]) == pytest.approx(info["scale"], abs=1e-12)
        assert bool(act.info["saturated"][e]) == info["saturated"]
    assert act.info["saturated"].any() and not act.info["saturated"].all()

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.assets.bluerov.controllers import (  # noqa: E402
    BLUEROV2_DAMPING,
    BatchStationKeeper,
    HoldPose,
    StationKeeper,
)


def random_states(rng, E):
    q = rng.normal(size=(E, 4))
    q[:, 0] += 3.0
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return (rng.normal(scale=0.8, size=(E, 3)), q, rng.normal(scale=0.3, size=(E, 3)),
            rng.normal(scale=0.5, size=(E, 3)))


def test_batch_matches_per_vehicle():
    rng = np.random.default_rng(0)
    E, dt = 9, 1.0 / 30.0
    kw = dict(ki=(25.0, 0.0, 40.0), i_clamp=6.0)
    batch = BatchStationKeeper(E, **kw)
    singles = [StationKeeper(**kw) for _ in range(E)]
    target = rng.normal(size=(E, 3))
    yaw_t = rng.uniform(-3.1, 3.1, size=E)
    for _ in range(60):
        pos, q, v, w = random_states(rng, E)
        wb, _ = batch.command(pos_w=pos, quat_w=q, lin_vel_b=v, ang_vel_b=w,
                              target_pos=target, target_yaw=yaw_t, dt=dt)
        for e, k in enumerate(singles):
            ws, _ = k.command(pos_w=pos[e], quat_w=q[e], lin_vel_b=v[e], ang_vel_b=w[e],
                              target_pos=target[e], target_yaw=yaw_t[e], dt=dt)
            assert np.allclose(wb[e].numpy(), ws, rtol=1e-12, atol=1e-10)


def test_reset_clears_one_environment():
    k = BatchStationKeeper(3)
    pos = np.zeros((3, 3))
    q = np.tile([1.0, 0, 0, 0], (3, 1))
    for _ in range(5):
        k.command(pos_w=pos, quat_w=q, lin_vel_b=pos, ang_vel_b=pos,
                  target_pos=(1.0, 0.0, 0.0), target_yaw=0.0, dt=0.1)
    k.reset(torch.tensor([1]))
    assert float(k._i[1].abs().sum()) == 0.0 and float(k._i[0].abs().sum()) > 0.0


def test_hold_pose_batched():
    E = 4
    hp = HoldPose(BatchStationKeeper(E), target_pos=torch.zeros(E, 3))
    odom = torch.zeros(E, 13, dtype=torch.float64)
    odom[:, 3] = 1.0
    odom[:, 0] = torch.arange(E, dtype=torch.float64)
    out = hp({"odom": odom}, 0.05)["thrusters"]
    assert out.shape == (E, 6)
    assert float(out[0, 0]) == 0.0 and bool((out[1:, 0] < 0).all())


def test_moving_target_feeds_its_drag_forward():
    k = StationKeeper(damping=BLUEROV2_DAMPING)
    s = np.sqrt(0.5)
    q = np.array([s, 0.0, 0.0, s])
    wrench, _ = k.command(pos_w=np.zeros(3), quat_w=q, lin_vel_b=np.array([0.3, -0.1, 0.0]),
                          ang_vel_b=np.array([0.0, 0.0, 0.2]), target_pos=np.zeros(3),
                          target_yaw=np.pi / 2, dt=0.01, target_vel=np.array([0.1, 0.3, 0.0]),
                          target_yaw_rate=0.2)
    lin, quad = np.asarray(BLUEROV2_DAMPING)
    ref = np.array([0.3, -0.1, 0.0, 0.2])
    drag = lin * ref + quad * np.abs(ref) * ref
    assert np.allclose(wrench[[0, 1, 2, 5]], drag, atol=1e-9)

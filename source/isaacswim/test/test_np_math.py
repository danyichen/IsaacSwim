from __future__ import annotations

import math

import numpy as np
import pytest

from isaacswim.np_math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_axis_angle,
    quat_from_to,
    quat_mul,
    quat_to_mat,
    quat_to_rpy,
    rpy_to_mat,
    rpy_to_quat,
    wrap_pi,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_rpy(rng, n=1):
    out = rng.uniform(-math.pi, math.pi, (n, 3))
    out[:, 1] = rng.uniform(-1.4, 1.4, n)
    return out


def test_quat_mat_and_rpy_are_consistent(rng):
    for rpy in random_rpy(rng, 200):
        q = rpy_to_quat(rpy)
        v = rng.normal(size=3)
        assert np.allclose(quat_to_mat(q), rpy_to_mat(rpy), atol=1e-12)
        assert np.allclose(quat_to_mat(q) @ v, quat_apply(q, v), atol=1e-12)
        assert np.allclose(quat_apply_inverse(q, quat_apply(q, v)), v, atol=1e-12)
        assert np.allclose(np.array(quat_to_rpy(q)), rpy, atol=1e-9)


def test_rpy_to_mat_matches_alpha_ik(rng):
    def original(rpy):
        r, p, y = (float(v) for v in rpy)
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                  math.sin(p), math.cos(y), math.sin(y))
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    for rpy in rng.uniform(-3.0, 3.0, (50, 3)):
        assert np.allclose(rpy_to_mat(rpy), original(rpy), atol=0.0)


def test_quat_mul_applies_second_argument_first(rng):
    a, b = rpy_to_quat(rng.normal(size=3)), rpy_to_quat(rng.normal(size=3))
    v = rng.normal(size=3)
    assert np.allclose(quat_apply(quat_mul(a, b), v), quat_apply(a, quat_apply(b, v)))


def test_axis_angle_matches_rodrigues(rng):
    def rodrigues(axis, angle):
        k = np.asarray(axis, dtype=float)
        k = k / np.linalg.norm(k)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    for axis, angle in zip(rng.normal(size=(50, 3)), rng.uniform(-3.0, 3.0, 50)):
        assert np.allclose(quat_to_mat(quat_from_axis_angle(axis, angle)),
                           rodrigues(axis, angle))


def test_quat_from_to_maps_a_onto_b(rng):
    for a, b in zip(rng.normal(size=(50, 3)), rng.normal(size=(50, 3))):
        got = quat_apply(quat_from_to(a, b), a / np.linalg.norm(a))
        assert np.allclose(got, b / np.linalg.norm(b), atol=1e-9)


def test_quat_from_to_degenerate_cases():
    u = np.array([0.3, -0.5, 0.81])
    u /= np.linalg.norm(u)
    assert np.allclose(quat_apply(quat_from_to(u, u), u), u, atol=1e-12)
    assert np.allclose(quat_apply(quat_from_to(u, -u), u), -u, atol=1e-9)
    assert np.allclose(quat_from_to(np.zeros(3), u), [1.0, 0.0, 0.0, 0.0])


def test_wrap_pi_takes_the_short_way():
    assert wrap_pi(math.radians(359)) == pytest.approx(math.radians(-1))
    assert wrap_pi(math.radians(181)) == pytest.approx(math.radians(-179))
    assert wrap_pi(math.pi) == pytest.approx(-math.pi)
    assert wrap_pi(-math.pi) == pytest.approx(-math.pi)


def test_quat_to_rpy_survives_a_non_unit_quaternion():
    q = np.array([0.7071068, 0.0, 0.7071068, 0.0]) * 1.0000001
    assert not any(math.isnan(v) for v in quat_to_rpy(q))


def test_agrees_with_meshvolume_and_torch_math(rng):
    from isaacswim.meshvolume import quat_apply as mv_apply
    from isaacswim.meshvolume import quat_mul as mv_mul

    a, b = rpy_to_quat(rng.normal(size=3)), rpy_to_quat(rng.normal(size=3))
    v = rng.normal(size=3)
    assert np.allclose(quat_mul(a, b), mv_mul(a, b))
    assert np.allclose(quat_apply(a, v), mv_apply(a, v))

    torch = pytest.importorskip("torch")
    from isaacswim import torch_math as T

    qt, bt, vt = torch.tensor(a), torch.tensor(b), torch.tensor(v)
    assert np.allclose(quat_apply(a, v), T.quat_apply(qt, vt).numpy(), atol=1e-12)
    assert np.allclose(quat_mul(a, b), T.quat_mul(qt, bt).numpy(), atol=1e-12)
    assert np.allclose(quat_to_mat(a), T.quat_to_mat(qt).numpy(), atol=1e-12)


def test_torch_mat_to_quat_round_trip():
    import torch

    from isaacswim.torch_math import mat_to_quat, quat_to_mat
    g = torch.Generator().manual_seed(0)
    q = torch.randn(500, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[:, :1] < 0, -q, q)
    q[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    back = mat_to_quat(quat_to_mat(q))
    assert torch.allclose(quat_to_mat(back), quat_to_mat(q), atol=1e-12)
    assert torch.allclose(back[1:], q[1:], atol=1e-12)

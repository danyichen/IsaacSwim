from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, HydroWrench, Medium, wrench_warp
from isaacswim.added_mass import AddedMass

DT = 1.0 / 200.0
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)


def heavy_frame(E=1):
    A = torch.diag(torch.tensor([3.0, 40.0, 25.0, 0.8, 0.2, 0.4], dtype=torch.float64))
    m = torch.tensor([13.0], dtype=torch.float64)
    I = torch.diag(torch.tensor([0.26, 0.23, 0.37], dtype=torch.float64)).unsqueeze(0)
    return A.unsqueeze(0), m, I, AddedMass(A.unsqueeze(0), m, I, DT, E, dtype=torch.float64)


def test_constant_force_gives_the_heavier_acceleration():
    A, m, I, am = heavy_frame()
    M = torch.diag(torch.tensor([13.0, 13.0, 13.0, 0.26, 0.23, 0.37], dtype=torch.float64))
    K = torch.tensor([10.0, 10.0, 10.0, 0.1, 0.1, 0.1], dtype=torch.float64)
    G = torch.tensor([0.0, 0.0, -30.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    vel = torch.zeros(1, 1, 6, dtype=torch.float64)
    q = IDENT.view(1, 1, 4)
    for _ in range(5):
        f, t = am.force(q, vel, K[:3].view(1, 1, 3), K[3:].view(1, 1, 3))
        a = torch.linalg.solve(M, K + G + torch.cat([f, t], -1).view(6))
        vel = vel + DT * a.view(1, 1, 6)
    want = torch.linalg.solve(M + A[0], K + G)
    assert torch.allclose(a, want, rtol=1e-9, atol=1e-12)


def test_stable_with_drag_and_large_added_mass():
    A, m, I, am = heavy_frame()
    M = torch.diag(torch.tensor([13.0, 13.0, 13.0, 0.26, 0.23, 0.37], dtype=torch.float64))
    c = torch.tensor([18.0, 21.0, 37.0, 1.5, 1.5, 1.5], dtype=torch.float64)
    push = torch.tensor([20.0, -15.0, 8.0, 0.5, -0.3, 0.2], dtype=torch.float64)
    q = IDENT.view(1, 1, 4)
    v_sim = torch.zeros(6, dtype=torch.float64)
    v_ref = torch.zeros(6, dtype=torch.float64)
    for step in range(int(20.0 / DT)):
        s = push * math.copysign(1.0, math.sin(2 * math.pi * step * DT / 3.0))
        K = s - c * v_sim * v_sim.abs()
        f, t = am.force(q, v_sim.view(1, 1, 6), K[:3].view(1, 1, 3), K[3:].view(1, 1, 3))
        v_sim = v_sim + DT * torch.linalg.solve(M, K + torch.cat([f, t], -1).view(6))
        v_ref = v_ref + DT * torch.linalg.solve(M + A[0], s - c * v_ref * v_ref.abs())
        assert torch.isfinite(v_sim).all()
    assert torch.allclose(v_sim, v_ref, atol=2e-3)


def test_joint_reaction_converges():
    m1, m2, a1 = 1.0, 4.0, 3.0
    A = torch.zeros(2, 6, 6, dtype=torch.float64)
    A[0, 0, 0] = a1
    m = torch.tensor([m1, m2], dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    am = AddedMass(A, m, I, DT, 1, dtype=torch.float64)
    q = IDENT.expand(1, 2, 4)
    K = torch.zeros(1, 2, 6, dtype=torch.float64)
    K[0, 0, 0], K[0, 1, 0] = 2.0, 3.0
    vel = torch.zeros(1, 2, 6, dtype=torch.float64)
    for _ in range(200):
        f, t = am.force(q, vel, K[..., :3], K[..., 3:])
        total = K[..., 0] + f[..., 0]
        a = float(total.sum()) / (m1 + m2)
        vel = vel.clone()
        vel[..., 0] += DT * a
    assert abs(a - 5.0 / (m1 + m2 + a1)) < 1e-9


def test_reset_forgets_the_teleport():
    A, m, I, am = heavy_frame(E=2)
    q = IDENT.expand(2, 1, 4)
    z = torch.zeros(2, 1, 3, dtype=torch.float64)
    am.force(q, torch.zeros(2, 1, 6, dtype=torch.float64), z, z)
    jump = torch.zeros(2, 1, 6, dtype=torch.float64)
    jump[:, 0, 1] = 5.0
    am.reset([0])
    f, _ = am.force(q, jump, z, z)
    assert float(f[0].abs().max()) == 0.0
    assert float(f[1].abs().max()) > 1.0


def test_added_mass_matrix_offset_and_rotation():
    rho = 1000.0
    med = Medium(density=rho, gravity=(0.0, 0.0, 0.0))
    semi = (0.3, 0.1, 0.05)
    rot = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    s0 = FluidShape(body="b", semiaxes=semi)
    s1 = FluidShape(body="b", semiaxes=semi, quat=rot, pos=(0.0, 0.0, 0.2))
    A0 = HydroWrench([s0], ["b"], med, backend="torch").added_mass_matrix()[0]
    A1 = HydroWrench([s1], ["b"], med, backend="torch").added_mass_matrix()[0]
    vm, vi = rho * np.asarray(s0.virtual_mass), rho * np.asarray(s0.virtual_inertia)
    assert np.allclose(np.diag(A0), np.r_[vm, vi])
    assert np.isclose(A1[0, 0], vm[1]) and np.isclose(A1[1, 1], vm[0])
    assert np.isclose(A1[4, 4], vi[0] + A1[0, 0] * 0.2 ** 2)
    assert np.isclose(A1[0, 4], A1[0, 0] * 0.2)
    assert np.allclose(A1, A1.T)


def test_a_pose_write_is_caught_without_reset():
    A, m, I, am = heavy_frame(E=2)
    q = IDENT.expand(2, 1, 4)
    z = torch.zeros(2, 1, 3, dtype=torch.float64)
    still = torch.zeros(2, 1, 6, dtype=torch.float64)
    am.force(q, still, z, z, pos_w=z)
    jump = still.clone()
    jump[:, 0, 1] = 5.0
    pos = torch.zeros(2, 1, 3, dtype=torch.float64)
    pos[0, 0, 0] = 2.0
    pos[1, 0, 1] = 5.0 * DT
    f, _ = am.force(q, jump, z, z, pos_w=pos)
    assert float(f[0].abs().max()) == 0.0
    assert float(f[1].abs().max()) > 1.0


@pytest.mark.skipif(not wrench_warp.available(), reason="NVIDIA Warp not installed")
@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_warp_matches_torch(device):
    rng = np.random.default_rng(0)
    E, B = 5, 2
    L = rng.normal(size=(B, 6, 6))
    A = torch.as_tensor(L @ L.transpose(0, 2, 1) + 6.0 * np.eye(6))
    m = torch.as_tensor(rng.uniform(2.0, 15.0, size=(E, B)))
    J = rng.normal(size=(E, B, 3, 3))
    I = torch.as_tensor(0.05 * J @ J.transpose(0, 1, 3, 2) + 0.1 * np.eye(3))
    com = 0.1 * rng.normal(size=(E, B, 3))
    ref = AddedMass(A, m, I, DT, E, device=device, body_com=com, backend="torch")
    fast = AddedMass(A, m, I, DT, E, device=device, body_com=com, backend="warp")
    assert ref.backend == "torch" and fast.backend == "warp"

    def r(*shape):
        return torch.as_tensor(rng.normal(size=shape), dtype=torch.float32, device=device)

    pos = r(E, B, 3)
    q = r(E, B, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    for step in range(40):
        vel, F, T = r(E, B, 6), 10.0 * r(E, B, 3), r(E, B, 3)
        pos = pos + vel[..., :3] * DT
        if step == 20:
            pos[1] += 1.0
        if step == 30:
            for am in (ref, fast):
                am.reset([0, 3])
        a, b = ref.force(q, vel, F, T, pos), fast.force(q, vel, F, T, pos)
        for x, y in zip(a, b):
            scale = max(float(x.abs().max()), 1e-6)
            assert float((x - y).abs().max()) <= 1e-4 * scale, (step, float((x - y).abs().max()))

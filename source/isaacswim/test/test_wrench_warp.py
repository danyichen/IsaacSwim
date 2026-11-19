from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, HydroWrench, Medium, wrench_warp
from isaacswim.torch_math import quat_to_mat

pytestmark = pytest.mark.skipif(not wrench_warp.available(), reason="NVIDIA Warp not installed")

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
BODIES = ["hull", "arm", "fin"]


def _quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def shapes(rng):
    return [
        FluidShape.box("hull", [0.3, 0.2, 0.1], mesh=True, pos=[0.05, 0.0, 0.02], quat=_quat(rng)),
        FluidShape.cylinder("hull", 0.06, 0.2, "x", mesh=True, pos=[-0.1, 0.15, 0.0]),
        FluidShape("arm", semiaxes=np.array([0.2, 0.03, 0.02]), pos=np.array([0.1, 0.0, 0.0]),
                   quat=_quat(rng)),
        FluidShape.sphere("arm", 0.04, pos=[0.3, 0.0, 0.0]),
        FluidShape("fin", semiaxes=np.array([0.15, 0.1, 0.005]), quat=_quat(rng),
                   coef=np.array([0.5, 0.25, 1.5, 1.0, 1.0])),
    ]


def state(rng, E, device, depth_spread=0.4):
    q = torch.as_tensor(rng.normal(size=(E, len(BODIES), 4)), dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True)
    lin = torch.as_tensor(rng.normal(size=(E, len(BODIES), 3)), dtype=torch.float32)
    ang = torch.as_tensor(2.0 * rng.normal(size=(E, len(BODIES), 3)), dtype=torch.float32)
    pos = torch.as_tensor(rng.normal(size=(E, len(BODIES), 3)), dtype=torch.float32)
    pos[..., 2] = torch.as_tensor(rng.uniform(-depth_spread, depth_spread, size=(E, len(BODIES))))
    return [t.to(device) for t in (q, lin, ang, pos)]


def pair(medium, device, seed=0, **kw):
    rng = np.random.default_rng(seed)
    s = shapes(rng)
    ref = HydroWrench(s, BODIES, medium, device=device, backend="torch", **kw)
    fast = HydroWrench(s, BODIES, medium, device=device, backend="warp", **kw)
    assert ref.backend == "torch" and fast.backend == "warp"
    return ref, fast


def close(a, b):
    for x, y in zip(a, b):
        scale = max(float(x.abs().max()), 1e-6)
        assert float((x - y).abs().max()) <= 2e-5 * scale + 1e-7, \
            f"max |diff| {float((x - y).abs().max()):.3e} at scale {scale:.3e}"


@pytest.mark.parametrize("device", DEVICES)
def test_deep_water(device):
    med = Medium(density=1025.0, viscosity=1e-3)
    ref, fast = pair(med, device)
    q, lin, ang, pos = state(np.random.default_rng(1), 16, device)
    close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))


@pytest.mark.parametrize("device", DEVICES)
def test_free_surface_cuts_meshes_and_ellipsoids(device):
    med = Medium(density=1025.0, viscosity=1e-3, surface_height=0.0)
    ref, fast = pair(med, device)
    for seed in range(4):
        q, lin, ang, pos = state(np.random.default_rng(10 + seed), 32, device)
        close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))


@pytest.mark.parametrize("device", DEVICES)
def test_deep_fast_path_matches_the_cut(device):
    med = Medium(density=1025.0, viscosity=1e-3, surface_height=0.0)
    ref, fast = pair(med, device)
    q, lin, ang, pos = state(np.random.default_rng(3), 8, device)
    pos[..., 2] -= 5.0
    close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))
    nosurf = HydroWrench(shapes(np.random.default_rng(0)), BODIES,
                         Medium(density=1025.0, viscosity=1e-3), device=device,
                         backend="torch")
    close(nosurf.compute(q, lin, ang, pos_w=pos), ref.compute(q, lin, ang, pos_w=pos))


@pytest.mark.parametrize("device", DEVICES)
def test_currents(device):
    q, lin, ang, pos = state(np.random.default_rng(4), 8, device)
    med = Medium(density=1025.0, viscosity=1e-3, current=(0.3, -0.2, 0.05))
    ref, fast = pair(med, device)
    close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))

    def field(x):
        return torch.stack([0.5 * x[..., 2], torch.zeros_like(x[..., 0]), 0.1 * x[..., 0]], -1)
    med = Medium(density=1025.0, viscosity=1e-3, surface_height=0.0)
    ref, fast = pair(med, device)
    close(ref.compute(q, lin, ang, pos_w=pos, current=field),
          fast.compute(q, lin, ang, pos_w=pos, current=field))


@pytest.mark.parametrize("device", DEVICES)
def test_moving_surface(device):
    med = Medium(density=1025.0, viscosity=1e-3, surface_height=0.0)
    ref, fast = pair(med, device)
    q, lin, ang, pos = state(np.random.default_rng(5), 16, device)

    def waves(x):
        return 0.15 * torch.sin(3.0 * x[..., 0]) * torch.cos(2.0 * x[..., 1])
    close(ref.compute(q, lin, ang, pos_w=pos, surface=waves),
          fast.compute(q, lin, ang, pos_w=pos, surface=waves))


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_centre_of_mass(device):
    E = 6
    com = torch.as_tensor(np.random.default_rng(6).normal(scale=0.05, size=(E, 3, 3)),
                          dtype=torch.float32)
    med = Medium(density=1025.0, viscosity=1e-3, surface_height=0.0)
    ref, fast = pair(med, device, body_com=com)
    q, lin, ang, pos = state(np.random.default_rng(7), E, device)
    close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))
    com2 = com.flip(0)
    ref.set_body_com(com2)
    fast.set_body_com(com2)
    close(ref.compute(q, lin, ang, pos_w=pos), fast.compute(q, lin, ang, pos_w=pos))
    R = quat_to_mat(q)
    lin_com = lin + torch.cross(ang, (R @ com2.to(device).unsqueeze(-1)).squeeze(-1), dim=-1)
    want = ref.compute(q, lin, ang, pos_w=pos)
    for w in (ref, fast):
        close(want, w.compute(q, lin_com, ang, pos_w=pos, vel_at_com=True))


def test_float64_stays_on_torch():
    w = HydroWrench([FluidShape.sphere("b", 0.1)], ["b"], dtype=torch.float64)
    assert w.backend == "torch"
    with pytest.raises(ValueError):
        HydroWrench([FluidShape.sphere("b", 0.1)], ["b"], dtype=torch.float64, backend="warp")

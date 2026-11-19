from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, HydroWrench, Medium
from isaacswim.torch_math import quat_apply

DT = torch.float64
G = 9.81
RHO = 1000.0

STILL = Medium(density=RHO, viscosity=0.0, gravity=(0.0, 0.0, -G))
SURFACE = Medium(density=RHO, viscosity=0.0, gravity=(0.0, 0.0, -G), surface_height=0.0)


def build(shapes, bodies, medium=STILL, **kw):
    return HydroWrench(shapes, bodies, medium, dtype=DT, **kw)


def at_rest(w, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    z = torch.zeros(1, 1, 3, dtype=DT)
    return w.compute(torch.tensor([[list(quat)]], dtype=DT), z, z,
                     pos_w=torch.tensor([[list(pos)]], dtype=DT))


def test_archimedes():
    r = 0.1
    w = build([FluidShape.sphere("b", r)], ["b"])
    F, T = at_rest(w)
    volume = 4.0 / 3.0 * np.pi * r ** 3
    assert F[0, 0, 2].item() == pytest.approx(RHO * volume * G, rel=1e-12)
    assert torch.allclose(F[0, 0, :2], torch.zeros(2, dtype=DT), atol=1e-15)
    assert torch.allclose(T, torch.zeros_like(T), atol=1e-15)


def test_buoyancy_is_off_when_gravity_is_zero():
    w = build([FluidShape.sphere("b", 0.1)], ["b"],
              medium=Medium(density=RHO, viscosity=0.0, gravity=(0.0, 0.0, 0.0)))
    assert not w.buoyancy
    F, _ = at_rest(w)
    assert torch.allclose(F, torch.zeros_like(F), atol=1e-15)


def test_explicit_volume_overrides_the_ellipsoid():
    w = build([FluidShape.sphere("b", 0.1, volume=0.02)], ["b"])
    assert at_rest(w)[0][0, 0, 2].item() == pytest.approx(RHO * 0.02 * G, rel=1e-12)


def test_drag_only_shape_has_no_buoyancy():
    w = build([FluidShape.sphere("b", 0.1, volume=0.0)], ["b"])
    assert not w.buoyancy


@pytest.mark.parametrize("z,expected", [
    (-1.0, 1.0),
    (-0.1, 1.0),
    (0.0, 0.5),
    (0.05, 0.15625),
    (0.1, 0.0),
    (1.0, 0.0),
])
def test_submerged_fraction_of_a_sphere(z, expected):
    r = 0.1
    w = build([FluidShape.sphere("b", r)], ["b"], medium=SURFACE)
    F, _ = at_rest(w, pos=(0.0, 0.0, z))
    full = RHO * (4.0 / 3.0 * np.pi * r ** 3) * G
    assert F[0, 0, 2].item() == pytest.approx(expected * full, rel=1e-12)


def test_submerged_fraction_matches_monte_carlo_for_a_rotated_ellipsoid():
    rng = np.random.default_rng(0)
    semi = np.array([0.30, 0.12, 0.07])
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    centre = np.array([0.0, 0.0, -0.02])
    surface_z = 0.0

    shape = FluidShape(body="b", semiaxes=semi, quat=q)
    w = build([shape], ["b"], medium=Medium(density=RHO, viscosity=0.0,
                                            gravity=(0.0, 0.0, -G),
                                            surface_height=surface_z))

    up_shape = quat_apply(
        torch.tensor(_conj(q), dtype=DT), torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    ).view(1, 1, 3)
    depth = torch.tensor([[surface_z - centre[2]]], dtype=DT)
    frac, cob_shape = w.submerged_fraction(depth, up_shape)

    n = 2_000_000
    y = rng.normal(size=(n, 3))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    y *= rng.random((n, 1)) ** (1.0 / 3.0)
    pts_shape = y * semi
    R = _quat_mat(q)
    with np.errstate(all="ignore"):
        pts_world = pts_shape @ R.T + centre
    wet = pts_world[:, 2] < surface_z

    assert frac.item() == pytest.approx(wet.mean(), abs=2e-3)
    centroid_shape = (pts_shape[wet]).mean(axis=0)
    assert np.allclose(cob_shape[0, 0].numpy(), centroid_shape, atol=2e-3)


def test_centre_of_buoyancy_rises_towards_the_surface():
    r = 0.2
    w = build([FluidShape.sphere("b", r)], ["b"], medium=SURFACE)
    up = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DT)
    frac, cob = w.submerged_fraction(torch.zeros(1, 1, dtype=DT), up)
    assert frac.item() == pytest.approx(0.5, rel=1e-12)
    assert cob[0, 0, 2].item() == pytest.approx(-3.0 * r / 8.0, rel=1e-12)


def test_a_bottom_heavy_float_rights_itself():
    w = build([FluidShape.sphere("float", 0.1, pos=[0.0, 0.0, 0.3])], ["float"],
              medium=SURFACE)

    def roll_moment(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        quat = (c, s, 0.0, 0.0)
        _, T = at_rest(w, pos=(0.0, 0.0, -0.3), quat=quat)
        return T[0, 0, 0].item()

    assert roll_moment(0.0) == pytest.approx(0.0, abs=1e-12)
    assert roll_moment(+0.3) < -1e-6, "heeling to starboard must push back to port"
    assert roll_moment(-0.3) > +1e-6
    assert abs(roll_moment(0.4)) > abs(roll_moment(0.2))


def test_drag_fades_out_above_the_surface():
    r = 0.1
    shapes = [FluidShape.sphere("b", r)]
    w = build(shapes, ["b"], medium=SURFACE)
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)

    deep = w.compute(quat, lin, ang, pos_w=torch.tensor([[[0.0, 0.0, -1.0]]], dtype=DT))[0]
    air = w.compute(quat, lin, ang, pos_w=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DT))[0]
    assert deep[0, 0, 0].item() < -1e-3
    assert air.abs().max().item() < 1e-15

    w2 = build(shapes, ["b"], medium=SURFACE, drag_scales_with_submersion=False)
    air2 = w2.compute(quat, lin, ang, pos_w=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DT))[0]
    assert air2[0, 0, 0].item() == pytest.approx(deep[0, 0, 0].item(), rel=1e-12)


def test_free_surface_requires_positions():
    w = build([FluidShape.sphere("b", 0.1)], ["b"], medium=SURFACE)
    z = torch.zeros(1, 1, 3, dtype=DT)
    with pytest.raises(ValueError, match="pos_w"):
        w.compute(torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT), z, z)


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_a_flat_surface_callable_is_the_medium_surface():
    shapes = [FluidShape.sphere("float", 0.1, pos=[0.0, 0.0, 0.3])]
    flat = build(shapes, ["float"], medium=Medium(density=RHO, viscosity=0.0,
                                                  gravity=(0.0, 0.0, -G),
                                                  surface_height=0.25))
    waves = build(shapes, ["float"])
    level = lambda c: torch.full(c.shape[:-1], 0.25, dtype=c.dtype)  # noqa: E731
    quat = torch.tensor([[[np.cos(0.2), np.sin(0.2), 0.0, 0.0]]], dtype=DT)
    lin = torch.tensor([[[0.3, -0.1, 0.2]]], dtype=DT)
    ang = torch.tensor([[[0.1, 0.4, -0.2]]], dtype=DT)
    pos = torch.tensor([[[0.1, 0.2, -0.1]]], dtype=DT)
    F0, T0 = flat.compute(quat, lin, ang, pos_w=pos)
    F1, T1 = waves.compute(quat, lin, ang, pos_w=pos, surface=level)
    assert torch.allclose(F0, F1, atol=1e-12) and torch.allclose(T0, T1, atol=1e-12)


def test_a_crest_lifts_and_a_trough_drops():
    r = 0.1
    w = build([FluidShape.sphere("b", r)], ["b"])
    z = torch.zeros(1, 1, 3, dtype=DT)
    q = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    p = torch.zeros(1, 1, 3, dtype=DT)
    lift = {h: w.compute(q, z, z, pos_w=p, surface=lambda c, h=h: torch.full(
        c.shape[:-1], h, dtype=c.dtype))[0][0, 0, 2].item() for h in (-r, 0.0, r)}
    full = RHO * 4.0 / 3.0 * np.pi * r ** 3 * G
    assert lift[-r] == pytest.approx(0.0, abs=1e-9)
    assert lift[0.0] == pytest.approx(full / 2, rel=1e-9)
    assert lift[r] == pytest.approx(full, rel=1e-9)


def test_a_surface_callable_requires_positions():
    w = build([FluidShape.sphere("b", 0.1)], ["b"])
    z = torch.zeros(1, 1, 3, dtype=DT)
    with pytest.raises(ValueError, match="pos_w"):
        w.compute(torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT), z, z,
                  surface=lambda c: c[..., 2])

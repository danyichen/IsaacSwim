from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, HydroWrench, Medium
from isaacswim.torch_math import quat_apply, quat_apply_inverse

DT = torch.float64

WATER = Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, 0.0))


def build(shapes, bodies, medium=WATER, **kw):
    return HydroWrench(shapes, bodies, medium, dtype=DT, **kw)


def rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_sphere_drag_is_the_textbook_quadratic_law():
    r, v = 0.1, 0.5
    w = build([FluidShape.sphere("b", r)], ["b"])
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[v, 0.0, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)
    F, T = w.compute(quat, lin, ang)

    quadratic = -WATER.density * v * 0.5 * np.pi * r ** 2 * v
    stokes = -WATER.viscosity * 3.0 * np.pi * (2.0 / 3.0 * 3 * r) * v
    assert F[0, 0, 0].item() == pytest.approx(quadratic + stokes, rel=1e-12)
    assert torch.allclose(F[0, 0, 1:], torch.zeros(2, dtype=DT), atol=1e-15)
    assert torch.allclose(T, torch.zeros_like(T), atol=1e-15)


def _sphere_drag_x(r, v, **kw):
    w = build([FluidShape.sphere("b", r)], ["b"], **kw)
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)
    lin = torch.tensor([[[v, 0.0, 0.0]]], dtype=DT)
    return w.compute(quat, lin, ang)[0][0, 0, 0].item()


def test_stokes_and_quadratic_drag_add():
    r, v = 0.05, 0.01
    stokes = -WATER.viscosity * 6.0 * np.pi * r * v
    quadratic = -WATER.density * v * 0.5 * np.pi * r ** 2 * v
    assert _sphere_drag_x(r, v) == pytest.approx(stokes + quadratic, rel=1e-12)


def test_projected_area_hits_mujocos_minval_floor_when_small_and_slow():
    r, v = 1e-3, 1e-4
    stokes = -WATER.viscosity * 6.0 * np.pi * r * v
    quadratic = -WATER.density * v * 0.5 * np.pi * r ** 2 * v

    clamped = _sphere_drag_x(r, v)
    assert clamped == pytest.approx(stokes + quadratic / 316.2, rel=0.05)

    exact = _sphere_drag_x(r, v, minval=0.0)
    assert exact == pytest.approx(stokes + quadratic, rel=1e-12)


def test_zero_velocity_gives_exactly_zero_and_no_nan():
    w = build([FluidShape(body="b", semiaxes=[0.05, 0.02, 0.008])], ["b"])
    z = torch.zeros(4, 1, 3, dtype=DT)
    quat = torch.tensor([[1.0, 0, 0, 0]], dtype=DT).expand(4, 1, 4).contiguous()
    F, T = w.compute(quat, z, z)
    assert torch.isfinite(F).all() and torch.isfinite(T).all()
    assert torch.allclose(F, torch.zeros_like(F)) and torch.allclose(T, torch.zeros_like(T))


def test_wrench_is_frame_covariant():
    rng = np.random.default_rng(3)
    shape = FluidShape(body="b", semiaxes=[0.05, 0.02, 0.008],
                       pos=[0.01, -0.02, 0.03], quat=rand_quat(rng, 1)[0])
    w = build([shape], ["b"])

    v0 = torch.tensor(rng.normal(scale=0.4, size=(1, 1, 3)), dtype=DT)
    w0 = torch.tensor(rng.normal(scale=3.0, size=(1, 1, 3)), dtype=DT)
    ident = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    F0, T0 = w.compute(ident, v0, w0)

    for q in rand_quat(rng, 8):
        qt = torch.tensor(q, dtype=DT).view(1, 1, 4)
        F, T = w.compute(qt, quat_apply(qt, v0), quat_apply(qt, w0))
        assert torch.allclose(F, F0, atol=1e-12), "force is not frame covariant"
        assert torch.allclose(T, T0, atol=1e-12), "torque is not frame covariant"


def test_offset_shape_produces_the_moment_arm():
    arm = np.array([0.0, 0.3, 0.0])
    at_origin = build([FluidShape.sphere("b", 0.05)], ["b"])
    offset = build([FluidShape.sphere("b", 0.05, pos=arm)], ["b"])

    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[0.4, 0.0, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)

    F0, T0 = at_origin.compute(quat, lin, ang)
    F1, T1 = offset.compute(quat, lin, ang)
    assert torch.allclose(F0, F1, atol=1e-14), "translating a shape changed the force"
    expect = np.cross(arm, F0[0, 0].numpy())
    assert np.allclose(T1[0, 0].numpy(), expect, atol=1e-14)
    assert torch.allclose(T0, torch.zeros_like(T0), atol=1e-15)


def test_body_com_shifts_the_moment_not_the_force():
    com = np.array([[0.0, 0.0, 0.1]])
    a = build([FluidShape.sphere("b", 0.05)], ["b"])
    b = build([FluidShape.sphere("b", 0.05)], ["b"], body_com=com)

    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[0.0, 0.4, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)
    Fa, Ta = a.compute(quat, lin, ang)
    Fb, Tb = b.compute(quat, lin, ang)
    assert torch.allclose(Fa, Fb, atol=1e-14)
    expect = np.cross(-com[0], Fa[0, 0].numpy())
    assert np.allclose((Tb - Ta)[0, 0].numpy(), expect, atol=1e-14)


def test_rotation_about_the_shape_centre_uses_the_lever_velocity():
    w = build([FluidShape.sphere("b", 0.03, pos=[0.0, 0.2, 0.0])], ["b"])
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.zeros(1, 1, 3, dtype=DT)
    ang = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=DT)
    F, T = w.compute(quat, lin, ang)
    assert F[0, 0, 0].item() > 0.0
    assert T[0, 0, 2].item() < 0.0


def test_uniform_current_is_galilean():
    c = np.array([0.3, -0.1, 0.05])
    rng = np.random.default_rng(5)
    shape = FluidShape(body="b", semiaxes=[0.06, 0.02, 0.03], pos=[0.01, 0.0, -0.02])
    still = build([shape], ["b"])
    flowing = build([shape], ["b"], medium=Medium(density=1000.0, viscosity=0.001,
                                                  gravity=(0.0, 0.0, 0.0), current=tuple(c)))

    quat = torch.tensor(rand_quat(rng, 1), dtype=DT).view(1, 1, 4)
    lin = torch.tensor(rng.normal(scale=0.3, size=(1, 1, 3)), dtype=DT)
    ang = torch.tensor(rng.normal(scale=2.0, size=(1, 1, 3)), dtype=DT)

    F0, T0 = still.compute(quat, lin, ang)
    F1, T1 = flowing.compute(quat, lin + torch.tensor(c, dtype=DT), ang)
    assert torch.allclose(F0, F1, atol=1e-12)
    assert torch.allclose(T0, T1, atol=1e-12)


def test_current_field_callable_sees_shape_centres():
    seen = {}

    def shear(pos_w):
        seen["pos"] = pos_w.clone()
        return torch.stack([pos_w[..., 2] * 2.0,
                            torch.zeros_like(pos_w[..., 2]),
                            torch.zeros_like(pos_w[..., 2])], dim=-1)

    w = build([FluidShape.sphere("b", 0.05, pos=[0.0, 0.0, 0.5])], ["b"])
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    z = torch.zeros(1, 1, 3, dtype=DT)
    pos = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DT)
    F, _ = w.compute(quat, z, z, pos_w=pos, current=shear)

    assert seen["pos"][0, 0, 2].item() == pytest.approx(1.5)
    assert F[0, 0, 0].item() > 0.0


def test_shapes_accumulate_onto_their_bodies():
    one = build([FluidShape.sphere("a", 0.05)], ["a", "b"])
    two = build([FluidShape.sphere("a", 0.05), FluidShape.sphere("a", 0.05)], ["a", "b"])
    split = build([FluidShape.sphere("a", 0.05), FluidShape.sphere("b", 0.05)], ["a", "b"])

    quat = torch.tensor([[1.0, 0, 0, 0]], dtype=DT).expand(1, 2, 4).contiguous()
    lin = torch.tensor([[[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 2, 3, dtype=DT)

    F1 = one.compute(quat, lin, ang)[0]
    F2 = two.compute(quat, lin, ang)[0]
    F3 = split.compute(quat, lin, ang)[0]
    assert torch.allclose(F2[:, 0], 2.0 * F1[:, 0], atol=1e-14)
    assert torch.allclose(F3[:, 0], F1[:, 0], atol=1e-14)
    assert torch.allclose(F3[:, 1], torch.zeros(1, 3, dtype=DT), atol=1e-15)


def test_environments_are_independent():
    rng = np.random.default_rng(7)
    E = 16
    w = build([FluidShape(body="b", semiaxes=[0.05, 0.02, 0.03], pos=[0.01, 0.0, 0.0])], ["b"])
    quat = torch.tensor(rand_quat(rng, E), dtype=DT).view(E, 1, 4)
    lin = torch.tensor(rng.normal(scale=0.3, size=(E, 1, 3)), dtype=DT)
    ang = torch.tensor(rng.normal(scale=2.0, size=(E, 1, 3)), dtype=DT)
    F, T = w.compute(quat, lin, ang)
    for e in (0, 3, 7, 15):
        Fe, Te = w.compute(quat[e:e + 1], lin[e:e + 1], ang[e:e + 1])
        assert torch.allclose(F[e:e + 1], Fe, atol=1e-14)
        assert torch.allclose(T[e:e + 1], Te, atol=1e-14)


def test_wrench_never_does_work_on_the_body():
    rng = np.random.default_rng(11)
    shapes = [FluidShape(body="b", semiaxes=[0.05, 0.02, 0.008], pos=[0.01, 0.0, 0.0],
                         quat=rand_quat(rng, 1)[0])]
    w = build(shapes, ["b"])
    E = 256
    quat = torch.tensor(rand_quat(rng, E), dtype=DT).view(E, 1, 4)
    lin = torch.tensor(rng.normal(scale=0.5, size=(E, 1, 3)), dtype=DT)
    ang = torch.tensor(rng.normal(scale=5.0, size=(E, 1, 3)), dtype=DT)
    F, T = w.compute(quat, lin, ang)

    v_l = quat_apply_inverse(quat, lin)
    w_l = quat_apply_inverse(quat, ang)
    power = (F * v_l).sum(-1) + (T * w_l).sum(-1)
    assert power.max().item() <= 1e-9, f"fluid did work on the body: {power.max().item():.3e}"


def test_medium_swap_rescales_by_density():
    shape = FluidShape(body="b", semiaxes=[0.05, 0.02, 0.03])
    w = build([shape], ["b"])
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[0.3, 0.1, 0.0]]], dtype=DT)
    ang = torch.tensor([[[0.0, 1.0, 0.5]]], dtype=DT)
    F0, T0 = w.compute(quat, lin, ang)

    w.set_medium(Medium(density=2000.0, viscosity=0.002, gravity=(0.0, 0.0, 0.0)))
    F1, T1 = w.compute(quat, lin, ang)
    assert torch.allclose(F1, 2.0 * F0, rtol=1e-12)
    assert torch.allclose(T1, 2.0 * T0, rtol=1e-12)


def test_unknown_body_is_rejected_with_the_body_list():
    with pytest.raises(KeyError, match="thruster"):
        build([FluidShape.sphere("thruster", 0.1)], ["hull", "fin"])


def test_float32_stays_close_to_float64():
    shape = FluidShape(body="b", semiaxes=[0.05, 0.02, 0.008], pos=[0.01, 0.0, 0.0])
    w64 = HydroWrench([shape], ["b"], WATER, dtype=torch.float64)
    w32 = HydroWrench([shape], ["b"], WATER, dtype=torch.float32)
    quat = torch.tensor([[[1.0, 0, 0, 0]]])
    lin = torch.tensor([[[0.3, -0.2, 0.1]]])
    ang = torch.tensor([[[1.0, 0.5, -2.0]]])
    F64, T64 = w64.compute(quat.double(), lin.double(), ang.double())
    F32, T32 = w32.compute(quat.float(), lin.float(), ang.float())
    assert torch.allclose(F32.double(), F64, rtol=1e-5, atol=1e-9)
    assert torch.allclose(T32.double(), T64, rtol=1e-5, atol=1e-9)

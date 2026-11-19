from __future__ import annotations

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco", reason="MuJoCo parity needs the mujoco package")
if tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 14):
    pytest.skip(f"parity is against MuJoCo >= 3.14 (installed: {mujoco.__version__})",
                allow_module_level=True)

from isaacswim import FluidShape, HydroWrench  # noqa: E402
from isaacswim.mjcf import shapes_from_mjmodel  # noqa: E402

DT = torch.float64
TOL = 1e-11

XML = """
<mujoco>
  <option timestep="0.001" integrator="Euler" gravity="0 0 0"
          density="{rho}" viscosity="{mu}" wind="{wind}"/>
  <worldbody>
    <body name="b" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="g" type="ellipsoid" size="{a} {b} {c}" pos="{gp}" quat="{gq}"
            fluidshape="ellipsoid"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_model(semi, *, rho=1000.0, mu=0.001, gpos=(0, 0, 0), gquat=(1, 0, 0, 0),
               wind=(0, 0, 0)):
    return mujoco.MjModel.from_xml_string(XML.format(
        a=semi[0], b=semi[1], c=semi[2], rho=rho, mu=mu,
        gp=" ".join(str(x) for x in gpos),
        gq=" ".join(str(x) for x in gquat),
        wind=" ".join(str(x) for x in wind)))


def mujoco_wrench(model, data, v, w):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:3] = v
    data.qvel[3:6] = w
    mujoco.mj_forward(model, data)
    return data.qfrc_passive[:3].copy(), data.qfrc_passive[3:6].copy()


def isaacswim_wrench(model, v, w):
    shapes, medium = shapes_from_mjmodel(model)
    medium.gravity = (0.0, 0.0, 0.0)
    hydro = HydroWrench(shapes, ["b"], medium, dtype=DT)
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DT)
    F, T = hydro.compute(quat,
                         torch.tensor(v, dtype=DT).view(1, 1, 3),
                         torch.tensor(w, dtype=DT).view(1, 1, 3),
                         pos_w=torch.zeros(1, 1, 3, dtype=DT))
    return F[0, 0].numpy(), T[0, 0].numpy()


def states():
    rng = np.random.default_rng(0)
    out = []
    for k in range(3):
        for s in (+1.0, -1.0):
            e = np.zeros(3)
            e[k] = s * 0.3
            out.append((e, np.zeros(3)))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 2.0
        out.append((np.zeros(3), e))
    out.append((np.array([1e-4, 0.0, 0.0]), np.zeros(3)))
    out.append((np.zeros(3), np.zeros(3)))
    for _ in range(24):
        out.append((rng.normal(scale=0.4, size=3), rng.normal(scale=3.0, size=3)))
    return out


def compare(model, label):
    data = mujoco.MjData(model)
    worst_f = worst_t = 0.0
    for v, w in states():
        fm, tm = mujoco_wrench(model, data, v, w)
        fh, th = isaacswim_wrench(model, v, w)
        ef = np.linalg.norm(fh - fm) / max(np.linalg.norm(fm), 1e-30)
        et = np.linalg.norm(th - tm) / max(np.linalg.norm(tm), 1e-30)
        assert ef < TOL, (f"{label}: force differs at v={v} w={w}\n"
                          f"  mujoco   {fm}\n  isaacswim {fh}\n  rel {ef:.3e}")
        assert et < TOL, (f"{label}: torque differs at v={v} w={w}\n"
                          f"  mujoco   {tm}\n  isaacswim {th}\n  rel {et:.3e}")
        worst_f, worst_t = max(worst_f, ef), max(worst_t, et)
    return worst_f, worst_t


SEMIAXES = {
    "arm segment": (0.015290998, 0.022823444, 0.022823360),
    "hull": (0.095963920, 0.095963084, 0.189760924),
    "eccentric": (0.05, 0.02, 0.008),
    "near sphere": (0.1, 0.1, 0.1),
    "very slender": (0.4, 0.005, 0.005),
}


@pytest.mark.parametrize("label,semi", SEMIAXES.items())
def test_matches_mujoco_at_the_origin(label, semi):
    compare(make_model(semi), label)


@pytest.mark.parametrize("label,semi", SEMIAXES.items())
def test_matches_mujoco_with_an_offset_geom(label, semi):
    compare(make_model(semi, gpos=(0.031, -0.017, 0.043)), f"{label} offset")


def test_matches_mujoco_with_an_offset_and_rotated_geom():
    q = np.array([0.3, -0.5, 0.7, 0.41])
    q /= np.linalg.norm(q)
    compare(make_model(SEMIAXES["eccentric"], gpos=(0.02, 0.05, -0.03), gquat=q),
            "offset + rotated")


def test_matches_mujoco_in_a_different_medium():
    compare(make_model(SEMIAXES["hull"], rho=1025.0, mu=0.00122), "sea water")


def test_matches_mujocos_wind():
    model = make_model(SEMIAXES["eccentric"], wind=(0.2, -0.1, 0.05))
    shapes, medium = shapes_from_mjmodel(model)
    assert medium.current == pytest.approx((0.2, -0.1, 0.05))
    compare(model, "wind")


def test_derived_added_mass_agrees_with_mujocos_to_its_own_accuracy():
    model = make_model(SEMIAXES["eccentric"])
    imported, _ = shapes_from_mjmodel(model)
    derived, _ = shapes_from_mjmodel(model, recompute_added_mass=True)
    assert np.allclose(derived[0].virtual_mass, imported[0].virtual_mass, rtol=2e-4)
    assert np.allclose(derived[0].virtual_inertia, imported[0].virtual_inertia,
                       rtol=2e-4, atol=1e-18)


@pytest.mark.parametrize("geom,semi", [
    ('type="sphere" size="0.3"', (0.3, 0.3, 0.3)),
    ('type="capsule" size="0.1 0.4"', (0.1, 0.1, 0.5)),
    ('type="cylinder" size="0.1 0.4"', (0.1, 0.1, 0.4)),
    ('type="box" size="0.1 0.2 0.3"', (0.1, 0.2, 0.3)),
])
def test_primitive_shape_constructors_match_mujocos_convention(geom, semi):
    xml = f"""<mujoco><worldbody><body><freejoint/>
      <geom {geom} fluidshape="ellipsoid"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    ref = model.geom_fluid[0]

    shapes, _ = shapes_from_mjmodel(model, recompute_added_mass=True)
    assert np.allclose(shapes[0].semiaxes, semi, rtol=1e-12)

    built = FluidShape(body="b", semiaxes=semi)
    assert np.allclose(built.virtual_mass, ref[6:9], rtol=2e-4)
    assert np.allclose(built.virtual_inertia, ref[9:12], rtol=2e-4, atol=1e-18)


def test_two_geoms_on_one_body_sum_the_way_mujoco_sums_them():
    xml = """
    <mujoco>
      <option gravity="0 0 0" density="1000" viscosity="0.001"/>
      <worldbody><body name="b"><freejoint/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="ellipsoid" size="0.05 0.02 0.02" pos="0.1 0 0" fluidshape="ellipsoid"/>
        <geom type="ellipsoid" size="0.04 0.02 0.03" pos="-0.1 0.02 0" fluidshape="ellipsoid"/>
      </body></worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    shapes, medium = shapes_from_mjmodel(model)
    assert len(shapes) == 2
    medium.gravity = (0.0, 0.0, 0.0)
    hydro = HydroWrench(shapes, ["b"], medium, dtype=DT)

    rng = np.random.default_rng(4)
    for _ in range(16):
        v, w = rng.normal(scale=0.4, size=3), rng.normal(scale=3.0, size=3)
        fm, tm = mujoco_wrench(model, data, v, w)
        F, T = hydro.compute(torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT),
                             torch.tensor(v, dtype=DT).view(1, 1, 3),
                             torch.tensor(w, dtype=DT).view(1, 1, 3),
                             pos_w=torch.zeros(1, 1, 3, dtype=DT))
        assert np.allclose(F[0, 0].numpy(), fm, rtol=TOL, atol=1e-14)
        assert np.allclose(T[0, 0].numpy(), tm, rtol=TOL, atol=1e-14)


@pytest.mark.parametrize("half", [(0.20, 0.05, 0.10), (0.1, 0.1, 0.1), (0.3, 0.02, 0.05)])
def test_from_inertia_matches_mujoco_for_a_box_like_part(half):
    a, b, c = half
    v = np.array([[sx * a, sy * b, sz * c]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    verts = " ".join(f"{x:.9f}" for x in v.flatten())
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <asset><mesh name="m" vertex="{verts}"/></asset>
        <option gravity="0 0 0" density="1000" viscosity="0.001"/>
        <worldbody><body name="b"><freejoint/>
          <geom type="mesh" mesh="m" fluidshape="ellipsoid"/>
        </body></worldbody>
      </mujoco>""")

    mj_semi = np.sort(np.array(model.geom_size[0]))
    fitted = FluidShape.from_inertia("b", float(model.body_mass[1]),
                                     np.array(model.body_inertia[1]))
    assert np.allclose(np.sort(fitted.semiaxes), mj_semi, rtol=1e-9), (
        f"CAD import diverges from MuJoCo: mujoco {mj_semi}, isaacswim "
        f"{np.sort(fitted.semiaxes)}")

    mj_vol = 4.0 / 3.0 * np.pi * float(np.prod(mj_semi))
    assert fitted.volume == pytest.approx(mj_vol, rel=1e-9)


def _bare(opt, geom):
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><option {opt}/><worldbody><body name='b'><freejoint/>"
        f"<geom {geom}/></body></worldbody></mujoco>")


def test_fluid_model_detection():
    from isaacswim.mjcf import fluid_model_of

    box = "type='box' size='.05 .02 .008'"
    wet = "gravity='0 0 0' density='1000' viscosity='0.001'"
    assert fluid_model_of(_bare("gravity='0 0 0'", box)) == "none"
    assert fluid_model_of(_bare(wet, box)) == "inertia_box"
    assert fluid_model_of(_bare(wet, box + " fluidshape='ellipsoid'")) == "both"
    assert fluid_model_of(_bare("gravity='0 0 0' wind='1 0 0' viscosity='0.001'",
                                box)) == "inertia_box"


def test_ellipsoid_importer_refuses_an_inertia_box_model():
    model = _bare("gravity='0 0 0' density='1000' viscosity='0.001'",
                  "type='box' size='.05 .02 .008'")
    with pytest.raises(ValueError, match="INERTIA-BOX"):
        shapes_from_mjmodel(model)

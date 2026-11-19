from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="these compare against MuJoCo itself")

from isaacswim.usd import _principal_obb_of_mesh  # noqa: E402


def box_mesh(half):
    a, b, c = half
    pts = np.array([[sx * a, sy * b, sz * c]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    idx = np.array([i for q in quads for i in q], dtype=np.int64)
    counts = np.full(len(quads), 4, dtype=np.int64)
    return pts, counts, idx


def mujoco_mesh_semiaxes(pts):
    verts = " ".join(f"{x:.9f}" for x in np.asarray(pts).flatten())
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <asset><mesh name="m" vertex="{verts}"/></asset>
        <option gravity="0 0 0" density="1000" viscosity="0.001"/>
        <worldbody><body name="b"><freejoint/>
          <geom type="mesh" mesh="m" fluidshape="ellipsoid"/>
        </body></worldbody>
      </mujoco>""")
    return np.sort(np.array(model.geom_size[0]))


@pytest.mark.parametrize("half", [
    (0.20, 0.05, 0.10),
    (0.10, 0.10, 0.10),
    (0.30, 0.02, 0.05),
    (0.07, 0.06, 0.055),
])
def test_mesh_ellipsoid_matches_mujoco(half):
    pts, counts, idx = box_mesh(half)
    ours = np.sort(_principal_obb_of_mesh(pts, counts, idx)[0])
    theirs = mujoco_mesh_semiaxes(pts)
    assert np.allclose(ours, theirs, rtol=1e-6), (
        f"isaacswim {ours} vs MuJoCo {theirs} -- the two toolchains would give this "
        "CAD part different drag")


def test_octahedron_pins_the_rule_a_box_cannot():
    r = np.array([0.3, 0.1, 0.05])
    pts = np.array([[ r[0],0,0], [-r[0],0,0], [0, r[1],0],
                    [0,-r[1],0], [0,0, r[2]], [0,0,-r[2]]], dtype=float)
    tris = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    counts = np.full(len(tris), 3, dtype=np.int64)
    idx = np.array([i for t in tris for i in t], dtype=np.int64)

    ours = np.sort(_principal_obb_of_mesh(pts, counts, idx)[0])
    assert np.allclose(ours, mujoco_mesh_semiaxes(pts), rtol=1e-6)
    assert np.allclose(ours, np.sort(r), rtol=1e-6)

    assert float(np.prod(ours)) == pytest.approx(float(np.prod(r)), rel=1e-6)


def test_mesh_volume_and_inertia_are_exact_for_a_box():
    half = np.array([0.2, 0.05, 0.1])
    pts, counts, idx = box_mesh(half)
    fitted = _principal_obb_of_mesh(pts, counts, idx)[0]
    assert np.allclose(np.sort(fitted), np.sort(half), rtol=1e-12)


def test_mesh_fit_is_translation_invariant():
    half = (0.2, 0.05, 0.1)
    pts, counts, idx = box_mesh(half)
    a = _principal_obb_of_mesh(pts, counts, idx)[0]
    b = _principal_obb_of_mesh(pts + np.array([3.0, -1.5, 0.7]), counts, idx)[0]
    assert np.allclose(np.sort(a), np.sort(b), rtol=1e-9)


def test_degenerate_mesh_returns_none_rather_than_nan():
    pts = np.array([[0.,0,0], [1,0,0], [0,1,0]])
    counts = np.array([3], dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    assert _principal_obb_of_mesh(pts, counts, idx) is None


def test_instance_proxy_traversal_is_requested():
    import inspect

    import isaacswim.usd as u
    src = inspect.getsource(u)
    assert src.count("TraverseInstanceProxies") >= 2, (
        "both the body scan and the per-body geometry scan must traverse "
        "instance proxies")

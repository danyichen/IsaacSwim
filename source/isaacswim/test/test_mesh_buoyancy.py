from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, HydroWrench, Medium, TriMesh
from isaacswim.meshvolume import MeshDisplacement, mesh_volume_centroid

DT = torch.float64
G = 9.81
RHO = 1000.0

STILL = Medium(density=RHO, viscosity=0.0, gravity=(0.0, 0.0, -G))
SURFACE = Medium(density=RHO, viscosity=0.0, gravity=(0.0, 0.0, -G), surface_height=0.0)


def test_convex_hull_has_outward_faces_and_translation_invariant_volume():
    pytest.importorskip("scipy")
    box = TriMesh.box([0.3, 0.2, 0.1])
    offset = np.array([4.0, -7.0, 2.0])
    moved = TriMesh(box.points + offset, box.faces)
    hull = moved.convex_hull()
    assert hull.open_edges == 0
    assert hull.volume == pytest.approx(8 * 0.3 * 0.2 * 0.1, abs=1e-12)
    assert np.allclose(hull.centroid, offset, atol=1e-12)


def test_reversed_face_is_reported_as_inconsistent_winding():
    box = TriMesh.box([0.3, 0.2, 0.1])
    faces = box.faces.copy()
    faces[0] = faces[0, ::-1]
    with pytest.warns(UserWarning, match="inconsistently wound"):
        broken = TriMesh(box.points, faces)
    assert broken.open_edges == 3


def build(shapes, bodies, medium=STILL, **kw):
    return HydroWrench(shapes, bodies, medium, dtype=DT, **kw)


def at_rest(w, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    z = torch.zeros(1, 1, 3, dtype=DT)
    return w.compute(torch.tensor([[list(quat)]], dtype=DT), z, z,
                     pos_w=torch.tensor([[list(pos)]], dtype=DT))


def octahedron(semi=(0.3, 0.1, 0.05)):
    a, b, c = semi
    pts = np.array([[a, 0, 0], [-a, 0, 0], [0, b, 0], [0, -b, 0], [0, 0, c], [0, 0, -c]])
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return pts, faces


def test_closed_form_volumes():
    assert TriMesh.box([0.3, 0.2, 0.1]).volume == pytest.approx(0.048, rel=1e-14)
    cyl = TriMesh.cylinder(0.1, 0.25, "z", 512)
    assert cyl.volume == pytest.approx(2 * np.pi * 0.01 * 0.25, rel=3e-5)
    cap = TriMesh.capsule(0.1, 0.2, "z", 128, 32)
    assert cap.volume == pytest.approx(2 * np.pi * 0.01 * 0.2 + 4 / 3 * np.pi * 1e-3,
                                       rel=1e-3)
    sph = TriMesh.sphere(0.1, 5)
    assert sph.volume == pytest.approx(4 / 3 * np.pi * 1e-3, rel=1e-3)


def test_tessellation_only_ever_under_states():
    exact_cyl = 2 * np.pi * 0.01 * 0.25
    for n in (8, 16, 64, 256):
        v = TriMesh.cylinder(0.1, 0.25, "z", n).volume
        assert v < exact_cyl
    e16 = exact_cyl - TriMesh.cylinder(0.1, 0.25, "z", 16).volume
    e64 = exact_cyl - TriMesh.cylinder(0.1, 0.25, "z", 64).volume
    assert e16 / e64 == pytest.approx(16.0, rel=0.05)


@pytest.mark.parametrize("kind,args,exact,gain", [
    ("box", ([0.3, 0.2, 0.1],), 8 * 0.3 * 0.2 * 0.1, 6 / np.pi),
    ("cylinder", (0.1, 0.25), 2 * np.pi * 0.01 * 0.25, 1.5),
    ("capsule", (0.1, 0.1), 2 * np.pi * 0.01 * 0.1 + 4 / 3 * np.pi * 1e-3, 1.25),
    ("sphere", (0.1,), 4 / 3 * np.pi * 1e-3, 1.0),
])
def test_primitives_take_their_analytic_volume_with_mesh_true(kind, args, exact, gain):
    s = getattr(FluidShape, kind)("b", *args, mesh=True)
    assert s.volume == pytest.approx(exact, rel=1e-14)
    assert s.volume / s.ellipsoid_volume == pytest.approx(gain, rel=1e-12)
    assert (s.mesh is None) == (kind == "sphere")


def test_an_explicit_volume_still_wins_over_mesh_true():
    s = FluidShape.box("b", [0.3, 0.2, 0.1], mesh=True, volume=0.01)
    assert s.volume == 0.01
    assert s.mesh is not None


def test_primitives_are_closed():
    for m in (TriMesh.box([0.3, 0.2, 0.1]), TriMesh.cylinder(0.1, 0.25, "y", 24),
              TriMesh.capsule(0.1, 0.2, "x", 24, 6), TriMesh.sphere(0.1, 2)):
        assert m.is_watertight, m


def test_inside_out_winding_is_repaired_not_absolute_valued():
    pts, faces = octahedron()
    good = TriMesh(pts, faces)
    flipped = TriMesh(pts, faces[:, [0, 2, 1]])
    assert flipped.volume == pytest.approx(good.volume, rel=1e-14)
    off = np.array([0.0, 0.0, 0.02])
    assert TriMesh(pts + off, faces[:, [0, 2, 1]]).centroid == pytest.approx(off, abs=1e-14)


def test_an_open_mesh_warns():
    pts, faces = octahedron()
    with pytest.warns(UserWarning, match="boundary edges"):
        TriMesh(pts, faces[:-1])


def test_a_flat_sheet_is_rejected():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ValueError, match="encloses no volume"):
        TriMesh(pts, np.array([[0, 1, 2], [0, 2, 3]]), check_watertight=False)


def test_polygon_face_streams_are_fanned():
    h = 0.15
    quads = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                      [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    pts = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
                   dtype=float)
    m = TriMesh(pts, counts=np.full(6, 4), indices=quads.reshape(-1))
    assert m.volume == pytest.approx((2 * h) ** 3, rel=1e-14)


def _submerged(mesh, depth, up):
    md = MeshDisplacement([mesh], dtype=DT)
    f, c = md.submerged(torch.tensor([[depth]], dtype=DT),
                        torch.tensor([[list(up)]], dtype=DT))
    return f[0, 0].item(), c[0, 0].numpy()


@pytest.mark.parametrize("depth,frac,cz", [
    (0.20, 1.00, 0.000),
    (0.10, 1.00, 0.000),
    (0.05, 0.75, -0.025),
    (0.00, 0.50, -0.050),
    (-0.05, 0.25, -0.075),
    (-0.10, 0.00, 0.000),
    (-0.50, 0.00, 0.000),
])
def test_a_box_cut_horizontally(depth, frac, cz):
    m = TriMesh.box([0.3, 0.2, 0.1])
    f, c = _submerged(m, depth, (0.0, 0.0, 1.0))
    assert f == pytest.approx(frac, rel=1e-12, abs=1e-14)
    assert c == pytest.approx([0.0, 0.0, cz], abs=1e-12)


def test_a_sphere_mesh_reproduces_the_spherical_cap():
    r = 0.2
    m = TriMesh.sphere(r, 5)
    ell = HydroWrench([FluidShape.sphere("b", r)], ["b"], SURFACE, dtype=DT)
    for t in (-0.8, -0.4, 0.0, 0.35, 0.9):
        depth = t * r
        up = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DT)
        want_f, want_c = ell.submerged_fraction(torch.tensor([[depth]], dtype=DT), up)
        got_f, got_c = _submerged(m, depth, (0.0, 0.0, 1.0))
        assert got_f == pytest.approx(want_f.item(), rel=3e-3)
        assert got_c[2] == pytest.approx(want_c[0, 0, 2].item(), abs=3e-4)


def test_the_cut_matches_monte_carlo_at_random_attitudes():
    rng = np.random.default_rng(7)
    semi = np.array([0.30, 0.10, 0.05])
    pts, faces = octahedron(semi)
    m = TriMesh(pts, faces)

    n = 400_000
    q = rng.uniform(-semi, semi, size=(n, 3))
    inside = (np.abs(q) / semi).sum(axis=1) <= 1.0
    q = q[inside]
    assert m.volume == pytest.approx(4.0 / 3.0 * semi.prod(), rel=1e-14)

    ups = rng.normal(size=(12, 3))
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    md = MeshDisplacement([m], dtype=DT)
    depths = rng.uniform(-0.2, 0.2, size=12)
    frac, cob = md.submerged(torch.tensor(depths, dtype=DT).view(-1, 1),
                             torch.tensor(ups, dtype=DT).view(-1, 1, 3))
    for k in range(12):
        with np.errstate(all="ignore"):
            wet = (q @ ups[k]) <= depths[k]
        assert frac[k, 0].item() == pytest.approx(wet.mean(), abs=4e-3)
        if wet.sum() > 5000:
            assert cob[k, 0].numpy() == pytest.approx(q[wet].mean(axis=0), abs=3e-3)


def test_the_clip_is_stable_at_grazing_incidence():
    m = TriMesh.box([0.2, 0.2, 0.1])
    for eps in (0.0, 1e-15, -1e-15, 1e-9, -1e-9):
        f, _ = _submerged(m, 0.1 + eps, (0.0, 0.0, 1.0))
        assert f == pytest.approx(1.0, abs=1e-7)
        f, _ = _submerged(m, -0.1 + eps, (0.0, 0.0, 1.0))
        assert f == pytest.approx(0.0, abs=1e-7)
    up = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    f, _ = _submerged(m, 0.0, up)
    assert f == pytest.approx(0.5, rel=1e-12)


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(11)
    meshes = [TriMesh.cylinder(0.07, 0.2, "z", 24), TriMesh.box([0.1, 0.15, 0.05])]
    ups = rng.normal(size=(9, 2, 3))
    ups /= np.linalg.norm(ups, axis=-1, keepdims=True)
    d = torch.tensor(rng.uniform(-0.25, 0.25, size=(9, 2)), dtype=DT)
    u = torch.tensor(ups, dtype=DT)

    big = MeshDisplacement(meshes, dtype=DT).submerged(d, u)
    small = MeshDisplacement(meshes, dtype=DT, max_clip_triangles=1).submerged(d, u)
    mid = MeshDisplacement(meshes, dtype=DT, max_clip_triangles=100).submerged(d, u)
    for a, b, c in zip(big, small, mid, strict=True):
        assert torch.equal(a, b)
        assert torch.equal(a, c)
    assert 0.0 < float(big[0].mean()) < 1.0


def test_fraction_is_monotone_and_bounded():
    m = TriMesh.capsule(0.08, 0.2, "x", 32, 8)
    up = np.array([0.3, -0.5, 0.81])
    up /= np.linalg.norm(up)
    last = -1.0
    for depth in np.linspace(-0.4, 0.4, 60):
        f, _ = _submerged(m, float(depth), up)
        assert -1e-12 <= f <= 1.0 + 1e-12
        assert f >= last - 1e-12
        last = f


def test_a_meshed_box_displaces_the_box_not_the_ellipsoid():
    h = np.array([0.3, 0.2, 0.1])
    mesh = TriMesh.box(h)
    shape = FluidShape(body="b", semiaxes=h, mesh=mesh)
    assert shape.volume == pytest.approx(8 * h.prod(), rel=1e-14)
    assert shape.ellipsoid_volume == pytest.approx(4 / 3 * np.pi * h.prod(), rel=1e-14)

    F, _ = at_rest(build([shape], ["b"]))
    assert F[0, 0, 2].item() == pytest.approx(RHO * 8 * h.prod() * G, rel=1e-12)

    F0, _ = at_rest(build([FluidShape(body="b", semiaxes=h)], ["b"]))
    assert F[0, 0, 2].item() / F0[0, 0, 2].item() == pytest.approx(6 / np.pi, rel=1e-12)


def test_mesh_buoyancy_false_falls_back_to_the_ellipsoid_cut():
    h = np.array([0.3, 0.2, 0.1])
    shape = FluidShape(body="b", semiaxes=h, mesh=TriMesh.box(h))
    on = build([shape], ["b"], medium=SURFACE)
    off = build([shape], ["b"], medium=SURFACE, mesh_buoyancy=False)
    assert off.mesh_disp is None
    deep = (0.0, 0.0, -1.0)
    assert at_rest(on, deep)[0][0, 0, 2].item() == \
        pytest.approx(at_rest(off, deep)[0][0, 0, 2].item(), rel=1e-12)
    on_q = at_rest(on, (0.0, 0.0, -0.05))[0][0, 0, 2].item()
    off_q = at_rest(off, (0.0, 0.0, -0.05))[0][0, 0, 2].item()
    assert on_q == pytest.approx(RHO * G * 0.75 * 8 * h.prod(), rel=1e-12)
    assert off_q / (RHO * G * 8 * h.prod()) == pytest.approx(
        (2 + 3 * 0.5 - 0.5 ** 3) * 0.25, rel=1e-12)
    assert on_q / off_q == pytest.approx(0.75 / 0.84375, rel=1e-12)


def test_a_submerged_mesh_pushes_at_its_own_centroid():
    h = np.array([0.1, 0.1, 0.1])
    off = np.array([0.0, 0.07, 0.0])
    mesh = TriMesh.box(h)
    mesh = TriMesh(mesh.points + off, mesh.faces)
    w = build([FluidShape(body="b", semiaxes=h, mesh=mesh)], ["b"])
    F, T = at_rest(w)
    f_up = RHO * G * 8 * h.prod()
    assert F[0, 0, 2].item() == pytest.approx(f_up, rel=1e-12)
    assert T[0, 0, 0].item() == pytest.approx(0.07 * f_up, rel=1e-12)
    assert abs(T[0, 0, 1].item()) < 1e-12


def test_an_explicit_volume_rescales_the_mesh_without_flattening_it():
    h = np.array([0.2, 0.2, 0.4])
    shape = FluidShape(body="b", semiaxes=h, mesh=TriMesh.box(h), volume=0.010)
    w = build([shape], ["b"], medium=SURFACE)
    assert at_rest(w, (0.0, 0.0, -1.0))[0][0, 0, 2].item() == \
        pytest.approx(RHO * G * 0.010, rel=1e-12)
    assert at_rest(w, (0.0, 0.0, 0.0))[0][0, 0, 2].item() == \
        pytest.approx(RHO * G * 0.005, rel=1e-12)


def test_a_box_floats_at_the_draft_archimedes_says_it_should():
    h = np.array([0.30, 0.20, 0.10])
    mass = 0.4 * RHO * 8 * h.prod()

    def equilibrium(w):
        lo, hi = -0.2, 0.2
        for _ in range(200):
            z = 0.5 * (lo + hi)
            net = at_rest(w, (0.0, 0.0, z))[0][0, 0, 2].item() - mass * G
            lo, hi = (z, hi) if net > 0 else (lo, z)
        return 0.1 - 0.5 * (lo + hi)

    mesh = build([FluidShape.box("hull", h, mesh=True)], ["hull"], medium=SURFACE)
    ell = build([FluidShape.box("hull", h)], ["hull"], medium=SURFACE)
    assert equilibrium(mesh) == pytest.approx(0.4 * 0.2, abs=1e-9)
    assert equilibrium(ell) / (0.4 * 0.2) == pytest.approx(1.71, abs=0.01)


def test_a_meshed_hull_rights_itself():
    hull = TriMesh.box([0.4, 0.15, 0.08])
    w = build([FluidShape(body="hull", semiaxes=[0.4, 0.15, 0.08], mesh=hull,
                          pos=[0.0, 0.0, 0.25])], ["hull"], medium=SURFACE)

    def roll_moment(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        _, T = at_rest(w, pos=(0.0, 0.0, -0.25), quat=(c, s, 0.0, 0.0))
        return T[0, 0, 0].item()

    assert roll_moment(0.0) == pytest.approx(0.0, abs=1e-12)
    assert roll_moment(+0.25) < -1e-6
    assert roll_moment(-0.25) > +1e-6
    assert abs(roll_moment(0.4)) > abs(roll_moment(0.2))


def test_meshed_and_bare_shapes_mix_on_one_robot():
    h = np.array([0.1, 0.1, 0.1])
    shapes = [FluidShape(body="a", semiaxes=h, mesh=TriMesh.box(h)),
              FluidShape.sphere("b", 0.1)]
    w = build(shapes, ["a", "b"], medium=SURFACE)
    assert w.mesh_disp.has_mesh.tolist() == [True, False]

    z = torch.zeros(1, 2, 3, dtype=DT)
    quat = torch.tensor([[[1.0, 0, 0, 0]] * 2], dtype=DT)
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=DT)
    F, _ = w.compute(quat, z, z, pos_w=pos)
    assert F[0, 0, 2].item() == pytest.approx(RHO * G * 0.5 * 8 * h.prod(), rel=1e-12)
    assert F[0, 1, 2].item() == pytest.approx(
        RHO * G * 0.5 * 4 / 3 * np.pi * 0.1 ** 3, rel=1e-12)


def test_drag_fades_with_the_meshed_submersion():
    h = np.array([0.2, 0.2, 0.1])
    w = build([FluidShape(body="b", semiaxes=h, mesh=TriMesh.box(h))], ["b"],
              medium=SURFACE)
    quat = torch.tensor([[[1.0, 0, 0, 0]]], dtype=DT)
    lin = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=DT)
    ang = torch.zeros(1, 1, 3, dtype=DT)
    fx = lambda z: w.compute(  # noqa: E731
        quat, lin, ang, pos_w=torch.tensor([[[0.0, 0.0, z]]], dtype=DT))[0][0, 0, 0].item()
    assert fx(-1.0) < -1e-3
    assert fx(1.0) == pytest.approx(0.0, abs=1e-15)
    assert fx(0.0) == pytest.approx(0.5 * fx(-1.0), rel=1e-12)


def test_batches_over_environments_and_slots():
    h = np.array([0.1, 0.1, 0.1])
    shapes = [FluidShape(body="a", semiaxes=h, mesh=TriMesh.box(h)),
              FluidShape(body="b", semiaxes=h, mesh=TriMesh.box(h))]
    w = build(shapes, ["a", "b"], medium=SURFACE)
    E = 5
    zs = torch.tensor([-1.0, -0.05, 0.0, 0.05, 1.0], dtype=DT)
    quat = torch.tensor([[1.0, 0, 0, 0]], dtype=DT).expand(E, 2, 4).contiguous()
    z = torch.zeros(E, 2, 3, dtype=DT)
    pos = torch.zeros(E, 2, 3, dtype=DT)
    pos[:, 0, 2] = zs
    pos[:, 1, 2] = -zs
    F, _ = w.compute(quat, z, z, pos_w=pos)
    full = RHO * G * 8 * h.prod()
    want = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=DT) * full
    assert torch.allclose(F[:, 0, 2], want, rtol=1e-12, atol=1e-12)
    assert torch.allclose(F[:, 1, 2], want.flip(0), rtol=1e-12, atol=1e-12)


def test_from_mesh_keeps_mujocos_ellipsoid_and_takes_the_meshs_volume():
    pts, faces = octahedron((0.3, 0.1, 0.05))
    s = FluidShape.from_mesh("b", pts, faces)
    assert np.sort(s.semiaxes) == pytest.approx([0.05, 0.1, 0.3], rel=1e-12)
    assert s.volume == pytest.approx(4.0 / 3.0 * 0.3 * 0.1 * 0.05, rel=1e-12)
    assert s.ellipsoid_volume / s.volume == pytest.approx(np.pi, rel=1e-12)


def test_from_mesh_places_the_geometry_where_the_prim_was():
    rng = np.random.default_rng(3)
    pts, faces = octahedron((0.3, 0.1, 0.05))
    pts = pts + np.array([0.02, -0.05, 0.11])
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    prim_pos = np.array([0.4, -0.2, 0.05])

    s = FluidShape.from_mesh("b", pts, faces, pos=prim_pos, quat=q)
    R = _quat_mat(s.quat)
    back = s.mesh.points @ R.T + s.pos
    want = pts @ _quat_mat(q).T + prim_pos
    assert back == pytest.approx(want, abs=1e-12)

    vol, cen = mesh_volume_centroid(want, faces)
    assert s.pos == pytest.approx(cen, abs=1e-12)
    assert s.volume == pytest.approx(vol, rel=1e-12)


def test_from_mesh_applies_usd_scale():
    pts, faces = octahedron((0.3, 0.1, 0.05))
    s = FluidShape.from_mesh("b", pts, faces, scale=(2.0, 2.0, 2.0))
    assert s.volume == pytest.approx(8 * 4.0 / 3.0 * 0.3 * 0.1 * 0.05, rel=1e-12)
    assert np.sort(s.semiaxes) == pytest.approx([0.1, 0.2, 0.6], rel=1e-12)


def test_from_mesh_rejects_geometry_it_cannot_use():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ValueError, match="degenerate or encloses no volume"):
        FluidShape.from_mesh("b", pts, np.array([[0, 1, 2], [0, 2, 3]]))


def test_shapes_round_trip_through_a_dict_with_their_mesh():
    pts, faces = octahedron()
    a = FluidShape.from_mesh("b", pts, faces)
    b = FluidShape.from_dict(a.to_dict())
    assert b.volume == pytest.approx(a.volume, rel=1e-14)
    assert b.mesh.points == pytest.approx(a.mesh.points, abs=1e-14)


def _quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

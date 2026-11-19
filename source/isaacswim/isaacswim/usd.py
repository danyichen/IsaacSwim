"""Fluid shapes from USD geometry, one per collision prim, as MuJoCo does per geom."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from isaacswim.meshvolume import (
    TriMesh,
    principal_obb,
    quat_apply,
    quat_mul,
    triangulate,
)
from isaacswim.meshvolume import mat_to_quat as _mat_to_quat
from isaacswim.shapes import DEFAULT_COEF, FluidShape

__all__ = ["shapes_from_usd", "shapes_from_usd_stage"]

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _geometry_from_prim(prim, UsdGeom, *, capsule_segments=16, capsule_rings=4,
                        cylinder_segments=16):
    t = prim.GetTypeName()
    ZERO, IDENT = np.zeros(3), np.eye(3)

    if t == "Sphere":
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 0.0)
        return (np.full(3, r) if r > 0 else None), ZERO, IDENT, t, None, None

    if t in ("Capsule", "Cylinder"):
        g = UsdGeom.Capsule(prim) if t == "Capsule" else UsdGeom.Cylinder(prim)
        r = float(g.GetRadiusAttr().Get() or 0.0)
        h = float(g.GetHeightAttr().Get() or 0.0)
        axis = str(g.GetAxisAttr().Get() or "Z").upper()
        if r <= 0.0:
            return None, ZERO, IDENT, t, None, None
        s = np.full(3, r)
        k = _AXIS_INDEX.get(axis, 2)
        s[k] = h / 2.0 + (r if t == "Capsule" else 0.0)
        ax = "xyz"[k]
        if t == "Capsule":
            m = TriMesh.capsule(r, h / 2.0, ax, capsule_segments, capsule_rings,
                                check_watertight=False)
            vol = 2.0 * np.pi * r * r * (h / 2.0) + 4.0 / 3.0 * np.pi * r ** 3
        else:
            m = TriMesh.cylinder(r, h / 2.0, ax, cylinder_segments,
                                 check_watertight=False)
            vol = np.pi * r * r * h
        return s, ZERO, IDENT, t, (m.points, m.faces), float(vol)

    if t == "Cube":
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 0.0)
        if size <= 0.0:
            return None, ZERO, IDENT, t, None, None
        half = np.full(3, size / 2.0)
        m = TriMesh.box(half, check_watertight=False)
        return half, ZERO, IDENT, t, (m.points, m.faces), float(size ** 3)

    if t == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or counts is None or idx is None:
            return None, ZERO, IDENT, t, None, None
        pts = np.array(pts, dtype=np.float64)
        faces = triangulate(np.array(counts, dtype=np.int64),
                            np.array(idx, dtype=np.int64))
        r = principal_obb(pts, faces) if faces.size else None
        if r is None:
            return None, ZERO, IDENT, t, None, None
        half, centroid, R = r
        return half, centroid, R, t, (pts, faces), None

    return None, ZERO, IDENT, t, None, None


def _principal_obb_of_mesh(points: np.ndarray, counts: np.ndarray, indices: np.ndarray):
    faces = triangulate(counts, indices)
    if faces.size == 0:
        return None
    return principal_obb(np.asarray(points, dtype=np.float64), faces)


def _decompose(xf: np.ndarray):
    m = np.asarray(xf, dtype=np.float64)
    pos = m[3, :3].copy()
    basis = m[:3, :3].copy()
    scale = np.linalg.norm(basis, axis=1)
    scale[scale < 1e-30] = 1.0
    R = (basis.T / scale).T
    return pos, _mat_to_quat(R.T), scale


def shapes_from_usd_stage(stage, *, body_names: Sequence[str] | None = None,
                          rename: dict[str, str] | None = None,
                          coef=None, include_meshes: bool = True,
                          include_invisible: bool = True,
                          mesh_buoyancy: bool = True,
                          prefer_collision: bool = True,
                          verbose: bool = True) -> list[FluidShape]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shapes_from_usd needs the `pxr` USD bindings, which ship with Isaac "
            "Sim rather than with isaacswim.  Run this under Isaac Sim's "
            "interpreter, or use isaacswim.mjcf on the MuJoCo side instead."
        ) from exc

    rename = rename or {}
    keep = None if body_names is None else set(body_names)
    coef = np.array(DEFAULT_COEF if coef is None else coef, dtype=np.float64)
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    bodies = [p for p in stage.Traverse(Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise ValueError("this stage has no prim with UsdPhysics.RigidBodyAPI; "
                         "there are no rigid bodies to attach drag ellipsoids to")

    shapes: list[FluidShape] = []
    skipped: list[tuple[str, str]] = []
    open_meshes: list[tuple[str, int]] = []
    seen: set[str] = set()
    for body in bodies:
        bname = body.GetName()
        if keep is not None and bname not in keep:
            continue
        if bname in seen:
            continue
        seen.add(bname)
        body_xf = np.array(xf_cache.GetLocalToWorldTransform(body), dtype=np.float64)
        body_inv = np.linalg.inv(body_xf)

        prims = []
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for prim in it:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim != body:
                it.PruneChildren()
                continue
            if prim.IsA(UsdGeom.Gprim):
                prims.append(prim)
        if prefer_collision and any(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims):
            prims = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        for prim in prims:
            if not include_invisible:
                purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
                if purpose in ("guide", "proxy"):
                    continue

            path = str(prim.GetPath())
            semi, off_pos, off_rot, kind, geom, exact_vol = \
                _geometry_from_prim(prim, UsdGeom)
            if kind == "Mesh" and not include_meshes:
                skipped.append((path, "Mesh (pass include_meshes=True)"))
                continue
            if semi is None:
                skipped.append((path, f"{kind}: no ellipsoid equivalent"))
                continue

            rel = np.array(xf_cache.GetLocalToWorldTransform(prim),
                           dtype=np.float64) @ body_inv
            pos, quat, scale = _decompose(rel)
            semi = semi * scale
            pos = pos + quat_apply(quat, off_pos * scale)
            quat = quat_mul(quat, _mat_to_quat(off_rot))

            if np.any(semi <= 0.0):
                skipped.append((path, "non-positive semi-axis after scale"))
                continue

            mesh = volume = None
            if mesh_buoyancy and geom is not None:
                try:
                    mesh = TriMesh.from_shape_frame(
                        geom[0], geom[1], centre=off_pos, rotation=off_rot,
                        scale=scale, check_watertight=False, name=path)
                except ValueError as exc:
                    skipped.append((path, f"no mesh buoyancy: {exc}"[:90]))
                else:
                    if exact_vol is not None:
                        volume = float(exact_vol) * float(np.prod(scale))
                    if mesh.open_edges:
                        open_meshes.append((path, mesh.open_edges))

            shapes.append(FluidShape(body=rename.get(bname, bname), semiaxes=semi,
                                     pos=pos, quat=quat, coef=coef,
                                     mesh=mesh, volume=volume))

    if not shapes:
        raise ValueError(
            "no convertible geometry was found.  USD prims must be Sphere, "
            "Capsule, Cylinder or Cube (or Mesh with include_meshes=True) under a "
            f"prim with RigidBodyAPI.  Skipped: {skipped[:6]}")
    meshed = [s for s in shapes if s.mesh is not None]
    if open_meshes:
        worst = sorted(open_meshes, key=lambda x: -x[1])[:3]
        warnings.warn(
            f"{len(open_meshes)} of {len(meshed)} meshes are not closed, so their "
            "displaced volume is only as good as the hole is small.  Worst: "
            + ", ".join(f"{p.rsplit('/', 1)[-1]} ({n} edges)" for p, n in worst)
            + ".  Repair them, or give those shapes an explicit `volume`.",
            stacklevel=2)
    if verbose:
        print(f"[IsaacSwim] {len(shapes)} drag ellipsoids from USD geometry on "
              f"{len({s.body for s in shapes})} bodies "
              f"(of {len(seen)} considered), skipped {len(skipped)}")
        if meshed:
            ell = sum(s.ellipsoid_volume for s in meshed)
            true = sum(s.volume for s in meshed)
            print(f"[IsaacSwim] mesh buoyancy on {len(meshed)}/{len(shapes)} shapes "
                  f"({sum(s.mesh.num_faces for s in meshed)} triangles): displaces "
                  f"{true * 1e3:.2f} L where the ellipsoids alone gave "
                  f"{ell * 1e3:.2f} L ({true / max(ell, 1e-30):.2f}x)")
        for path, why in skipped[:10]:
            print(f"[IsaacSwim]   {path}: {why}")
        if len(skipped) > 10:
            print(f"[IsaacSwim]   ... and {len(skipped) - 10} more")
    return shapes


def shapes_from_usd(usd_path: str, **kwargs) -> list[FluidShape]:
    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shapes_from_usd needs the `pxr` USD bindings, which ship with Isaac Sim."
        ) from exc
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {usd_path}")
    return shapes_from_usd_stage(stage, **kwargs)


def shapes_from_articulation_usd(asset, **kwargs) -> list[FluidShape]:
    try:
        import isaaclab.sim.utils.stage as stage_utils
    except ImportError:  # pragma: no cover
        import omni.isaac.core.utils.stage as stage_utils
    kwargs.setdefault("body_names", list(asset.body_names))
    return shapes_from_usd_stage(stage_utils.get_current_stage(), **kwargs)

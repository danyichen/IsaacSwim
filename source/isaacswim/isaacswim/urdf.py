"""Fluid shapes and mesh buoyancy from a URDF."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from isaacswim.np_math import quat_apply, quat_mul, rpy_to_quat

__all__ = ["shapes_from_urdf", "load_obj", "MOVABLE_JOINTS"]

MOVABLE_JOINTS = {"revolute", "prismatic", "continuous", "floating", "planar"}


def load_obj(path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _origin(el) -> tuple[np.ndarray, np.ndarray]:
    o = el.find("origin") if el is not None else None
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None
                                       else "0 0 0").split()])
    rpy = (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()
    return xyz, rpy_to_quat(rpy)


def shapes_from_urdf(urdf_path, *, source: str = "visual", verbose: bool = True):
    from isaacswim.shapes import FluidShape

    if source not in ("visual", "collision"):
        raise ValueError(f"source must be 'visual' or 'collision', not {source!r}")

    root = ET.parse(urdf_path).getroot()

    parent_of = {}
    for j in root.findall("joint"):
        xyz, quat = _origin(j)
        parent_of[j.find("child").get("link")] = (j.find("parent").get("link"),
                                                   xyz, quat, j.get("type"))

    def to_owner(link):
        pos, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        node = link
        while node in parent_of:
            parent, j_xyz, j_quat, j_type = parent_of[node]
            if j_type in MOVABLE_JOINTS:
                break
            pos = j_xyz + quat_apply(j_quat, pos)
            quat = quat_mul(j_quat, quat)
            node = parent
        return node, pos, quat

    base = pathlib.Path(urdf_path).parent
    shapes, skipped = [], []
    for link in root.findall("link"):
        owner, l_pos, l_quat = to_owner(link.get("name"))
        for geom in link.findall(source):
            mesh_el = geom.find("geometry/mesh")
            if mesh_el is None:
                continue
            path = base / mesh_el.get("filename")
            if path.suffix.lower() != ".obj":
                skipped.append((path.name, "not OBJ -- convert it first"))
                continue
            scale = [float(v) for v in mesh_el.get("scale", "1 1 1").split()]

            v_xyz, v_quat = _origin(geom)
            pos = l_pos + quat_apply(l_quat, v_xyz)
            quat = quat_mul(l_quat, v_quat)

            pts, faces = load_obj(path)
            try:
                shapes.append(FluidShape.from_mesh(
                    owner, pts, faces, pos=pos, quat=quat, scale=scale,
                    check_watertight=False))
            except ValueError as exc:
                skipped.append((path.name, str(exc)[:60]))
    if verbose and skipped:
        print(f"  skipped {len(skipped)} meshes: {skipped[:3]}")
    return shapes

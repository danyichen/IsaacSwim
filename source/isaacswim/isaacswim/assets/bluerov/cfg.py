from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from isaacswim.flow import Uniform
from isaacswim.trim import TrimVolume, displaced_volume, solve_trim
from isaacswim.urdf import shapes_from_urdf

__all__ = ["SPAWN_Z", "SEABED_Z", "FOAM_Z", "FOAM_SEMIAXES", "ARM_HOME",
           "ARM_PATTERN", "FRAME_PARTS", "build_hydro", "frame_part_shapes"]

SPAWN_Z = 0.0

SEABED_Z = -4.0

ARM_PATTERN = r"alpha_.*"

FOAM_Z = 0.118
FOAM_SEMIAXES = (0.170, 0.150, 0.030)
FOAM_VOLUME = 2.0e-3

MAKEUP_Z = -0.030
MAKEUP_SEMIAXES = (0.215, 0.275, 0.045)

FRAME_MESH = "bluerov2_heavy.obj"
FRAME_PARTS = 20
FRAME_PART_MIN_SIZE = 0.03


def frame_part_shapes(urdf_path, n_parts: int = FRAME_PARTS, body: str = "base_link"):
    import hashlib
    import xml.etree.ElementTree as ET

    from isaacswim.shapes import FluidShape

    urdf_path = Path(urdf_path)
    cache = urdf_path.with_name(f"frame_parts_{n_parts}.json")
    el = next(m for m in ET.parse(urdf_path).getroot().iter("mesh")
              if m.get("filename", "").endswith(FRAME_MESH))
    mesh_path = urdf_path.parent / el.get("filename")
    key = {"mesh": FRAME_MESH, "scale": el.get("scale", "1 1 1"),
           "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest()}
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if isinstance(cached, dict) and cached.get("key") == key:
        rows = cached["rows"]
    else:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh", process=True)
        mesh.apply_scale([float(v) for v in el.get("scale", "1 1 1").split()])
        fitted = [FluidShape.from_mesh(body, np.asarray(p.vertices), np.asarray(p.faces),
                                       check_watertight=False)
                  for p in mesh.split(only_watertight=False)
                  if p.is_watertight and p.extents.max() >= FRAME_PART_MIN_SIZE]
        fitted.sort(key=lambda s: -float(np.prod(s.semiaxes)))
        rows = [{"semiaxes": np.asarray(s.semiaxes).tolist(), "pos": np.asarray(s.pos).tolist(),
                 "quat": np.asarray(s.quat).tolist()} for s in fitted[:n_parts]]
        cache.write_text(json.dumps({"key": key, "rows": rows}, indent=1))
    out = []
    for r in rows:
        s = FluidShape(body=body, volume=0.0, **r)
        s.coef = np.asarray(s.coef, dtype=float).copy()
        s.coef[3] = 0.0
        out.append(s)
    return out


def _frame_split(geom, urdf_path, n_parts, body="base_link"):
    on_body = [s for s in geom if s.body == body]
    frame = max(on_body, key=lambda s: float(np.prod(s.semiaxes)))
    buoy = copy.deepcopy(frame)
    buoy.coef = np.zeros(5)
    buoy.virtual_mass = np.zeros(3)
    buoy.virtual_inertia = np.zeros(3)
    return ([s for s in geom if s is not frame] + [buoy]
            + frame_part_shapes(urdf_path, n_parts, body))

ARM_HOME = {
    "alpha_axis_e": 2.85,
    "alpha_axis_d": 2.45,
    "alpha_axis_c": 1.80,
    "alpha_axis_b": 2.85,
    "alpha_standard_jaws_rs1_130_joint": 0.20,
    "alpha_standard_jaws_rs1_139_joint": 0.20,
    "camera_mount_joint": 0.0,
}


def _volume_by_body(shapes):
    out = {}
    for s in shapes:
        out[s.body] = out.get(s.body, 0.0) + s.volume
    return out


def _vehicle_com_in_base(robot):
    from isaacswim.isaaclab import _lab
    d = _lab.data(robot)
    masses = _lab.view_masses(robot)[0]
    pos = getattr(d, "body_com_pos_w", None)
    if pos is None:
        pos = d.body_pos_w
    w = masses.to(pos.device).unsqueeze(-1)
    com_w = (pos[0] * w).sum(0) / w.sum()
    return (com_w - d.root_pos_w[0]).detach().cpu().numpy()


def _shape_pos_in_base(robot, shape):
    import torch

    from isaacswim.isaaclab import _lab
    from isaacswim.torch_math import quat_apply

    d = _lab.data(robot)
    idx = robot.body_names.index(shape.body)
    link_pos = d.body_pos_w[0, idx]
    link_quat = d.body_quat_w[0, idx]

    offset = torch.as_tensor(np.asarray(shape.pos, dtype=np.float32),
                             device=link_pos.device, dtype=link_pos.dtype)
    world = link_pos + quat_apply(link_quat.unsqueeze(0), offset.unsqueeze(0))[0]
    return (world - d.root_pos_w[0]).detach().cpu().numpy()


def build_hydro(robot, urdf_path, *, level_trim=True, ballast=None,
                current=(0.0, 0.0, 0.0), surface_z=None, seabed_z=SEABED_Z,
                device="cpu", arm_volume_scale=1.0, extra_geom=(), surface_fn=None,
                frame_parts=FRAME_PARTS, added_mass=True, physics_dt=None):
    from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, _lab
    from isaacswim.medium import Medium

    medium = Medium.sea_water()
    if surface_z is not None:
        medium.surface_height = surface_z

    geom = shapes_from_urdf(urdf_path, verbose=False)
    if not geom:
        raise ValueError(f"no visual meshes found in {urdf_path}")
    if frame_parts:
        geom = _frame_split(list(geom), urdf_path, frame_parts)
    geom = list(geom) + list(extra_geom)

    if arm_volume_scale != 1.0:
        arm_re = re.compile(ARM_PATTERN)
        for s in geom:
            if arm_re.match(s.body):
                s.volume = float(s.volume) * arm_volume_scale

    masses = _lab.view_masses(robot).numpy()[0]
    total_mass = float(masses.sum())
    com_b = _vehicle_com_in_base(robot)

    geom_volume, geom_cob = displaced_volume(
        [s.volume for s in geom], [_shape_pos_in_base(robot, s) for s in geom])
    extra, trim = solve_trim(
        geom_volume=geom_volume, geom_cob=geom_cob, total_mass=total_mass,
        com=com_b, density=medium.density,
        foam=TrimVolume(FOAM_SEMIAXES, FOAM_Z, max_volume=FOAM_VOLUME),
        makeup=TrimVolume(MAKEUP_SEMIAXES, MAKEUP_Z),
        ballast=ballast, level_trim=level_trim)
    shapes = geom + extra

    current_fn = None
    if any(current):
        current_fn = Uniform(current, seabed_z=seabed_z, layer=1.5, device=device)

    hydro = Hydrodynamics(
        HydrodynamicsCfg(shapes=shapes, medium=medium,
                         buoyancy=True, current_fn=current_fn, surface_fn=surface_fn,
                         added_mass=added_mass, physics_dt=physics_dt),
        robot,
    )

    coms = _lab.view_coms(robot).numpy()[0]

    info = {
        "total_mass": total_mass, **trim,
        "n_geom": len(geom), "body_names": list(robot.body_names),
        "body_mass": masses, "body_com": coms[:, :3], "medium": medium,
        "per_body_volume": _volume_by_body(geom),
    }
    return hydro, info

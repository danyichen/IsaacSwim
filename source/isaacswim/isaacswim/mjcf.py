"""Fluid shapes and water properties from a MuJoCo model."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence

import numpy as np

from isaacswim.medium import Medium
from isaacswim.shapes import FluidShape

__all__ = ["shapes_from_mjmodel", "shapes_from_mjcf", "save_json", "load_json",
           "fluid_model_of"]

_FLUID_ENABLE, _FLUID_COEF, _FLUID_VMASS, _FLUID_VINERT = 0, slice(1, 6), slice(6, 9), slice(9, 12)


def shapes_from_mjmodel(model, *, body_names: Sequence[str] | None = None,
                        rename: dict[str, str] | None = None,
                        recompute_added_mass: bool = False,
                        include_disabled: bool = False):
    import mujoco

    n_geom = model.ngeom
    keep = None if body_names is None else set(body_names)
    rename = rename or {}

    shapes: list[FluidShape] = []
    for g in range(n_geom):
        fluid = model.geom_fluid[g]
        on = bool(fluid[_FLUID_ENABLE])
        if not on and not include_disabled:
            continue
        bid = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if keep is not None and bname not in keep:
            continue
        semi = np.array(model.geom_size[g], dtype=np.float64)
        if np.any(semi <= 0.0):
            semi = _equivalent_semiaxes(model, g)
        kw = {}
        if not recompute_added_mass:
            kw["virtual_mass"] = np.array(fluid[_FLUID_VMASS], dtype=np.float64)
            kw["virtual_inertia"] = np.array(fluid[_FLUID_VINERT], dtype=np.float64)
        shapes.append(FluidShape(
            body=rename.get(bname, bname),
            semiaxes=semi,
            pos=np.array(model.geom_pos[g], dtype=np.float64),
            quat=np.array(model.geom_quat[g], dtype=np.float64),
            coef=np.array(fluid[_FLUID_COEF], dtype=np.float64),
            enabled=on,
            **kw,
        ))

    medium = Medium(
        density=float(model.opt.density),
        viscosity=float(model.opt.viscosity),
        gravity=tuple(float(x) for x in model.opt.gravity),
        current=tuple(float(x) for x in model.opt.wind),
    )

    if not shapes and fluid_model_of(model) == "inertia_box":
        raise ValueError(
            f"this model has fluid enabled (density={model.opt.density:g}, "
            f"viscosity={model.opt.viscosity:g}) but no geom "
            "sets fluidshape=\"ellipsoid\", so MuJoCo is running the INERTIA-BOX "
            "fluid model, not the ellipsoid one.  Importing it as ellipsoid shapes "
            "would silently drop all of its hydrodynamics.  Set "
            "fluidshape=\"ellipsoid\" on the geoms you want modelled.")
    return shapes, medium


def fluid_model_of(model) -> str:
    active = (float(model.opt.density) != 0.0
              or float(model.opt.viscosity) != 0.0
              or bool((model.opt.wind != 0.0).any()))
    if not active:
        return "none"
    n_ellipsoid = int(sum(bool(model.geom_fluid[g][_FLUID_ENABLE])
                          for g in range(model.ngeom)))
    if n_ellipsoid == 0:
        return "inertia_box"
    return "both" if model.nbody > 1 else "ellipsoid"


def _equivalent_semiaxes(model, g: int) -> np.ndarray:
    import mujoco

    gtype = int(model.geom_type[g])
    size = np.array(model.geom_size[g], dtype=np.float64)
    T = mujoco.mjtGeom
    if gtype == T.mjGEOM_SPHERE:
        return np.full(3, size[0])
    if gtype == T.mjGEOM_CAPSULE:
        return np.array([size[0], size[0], size[1] + size[0]])
    if gtype == T.mjGEOM_CYLINDER:
        return np.array([size[0], size[0], size[1]])
    if gtype in (T.mjGEOM_BOX, T.mjGEOM_ELLIPSOID):
        return size
    raise ValueError(
        f"geom {g} has type {gtype}, which has no ellipsoid equivalent "
        "(mesh, plane, hfield...).  Give the semi-axes explicitly.")


def shapes_from_mjcf(path: str | pathlib.Path, **kwargs):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    return shapes_from_mjmodel(model, **kwargs)


def save_json(path: str | pathlib.Path, shapes: Sequence[FluidShape],
              medium: Medium, *, source: str | None = None) -> pathlib.Path:
    doc = {
        "isaacswim_format": 1,
        "source": source,
        "medium": {
            "density": medium.density,
            "viscosity": medium.viscosity,
            "gravity": list(medium.gravity),
            "current": list(medium.current),
            "surface_height": medium.surface_height,
        },
        "shapes": [s.to_dict() for s in shapes],
    }
    p = pathlib.Path(path)
    p.write_text(json.dumps(doc, indent=1))
    return p


def load_json(path: str | pathlib.Path):
    doc = json.loads(pathlib.Path(path).read_text())
    if doc.get("isaacswim_format") != 1:
        raise ValueError(f"{path}: not a isaacswim fluid spec")
    m = doc["medium"]
    medium = Medium(density=m["density"], viscosity=m["viscosity"],
                    gravity=tuple(m["gravity"]), current=tuple(m["current"]),
                    surface_height=m["surface_height"])
    return [FluidShape.from_dict(d) for d in doc["shapes"]], medium


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m isaacswim.mjcf",
        description="Extract the fluid geometry of a MuJoCo model to a isaacswim JSON spec.")
    ap.add_argument("mjcf", help="path to the MJCF (scene.xml, model.xml, ...)")
    ap.add_argument("out", help="path to write the JSON spec to")
    ap.add_argument("--recompute-added-mass", action="store_true",
                    help="derive the added mass from the semi-axes instead of "
                         "copying MuJoCo's (accurate, but no longer bit-identical)")
    args = ap.parse_args(argv)

    shapes, medium = shapes_from_mjmodel(
        __import__("mujoco").MjModel.from_xml_path(args.mjcf),
        recompute_added_mass=args.recompute_added_mass)
    save_json(args.out, shapes, medium, source=args.mjcf)

    bodies = sorted({s.body for s in shapes})
    total_v = sum(s.volume for s in shapes)
    print(f"{len(shapes)} fluid shapes on {len(bodies)} bodies -> {args.out}")
    print(f"medium: density {medium.density} kg/m^3, viscosity {medium.viscosity} Pa.s, "
          f"wind {medium.current}")
    print(f"displaced volume {total_v * 1e3:.3f} L "
          f"(= {total_v * medium.density:.3f} kg of fluid; a neutrally buoyant "
          f"robot has exactly this mass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

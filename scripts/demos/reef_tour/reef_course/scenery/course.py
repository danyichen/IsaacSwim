from __future__ import annotations

import numpy as np

from .. import course as T
from ..service import SERVICE_PIPE
from . import models as M
from . import props as P

FLAT_RADIUS = 13.0
PIPE_ALTITUDE = 1.0
ROOTS = ("/World/course_seabed", "/World/scenery", "/World/course")

REFLECTIVITY = (
    ("seabed", 1.0),
    ("concrete", 2.0),
    ("coat", 2.0),
    ("berm", 1.0),
    ("rock", 1.6),
    ("pebble", 1.2),
    ("sponge", 0.8),
    ("seaweed", 0.4),
    ("coral", 0.8),
    ("rope", 0.5),
    ("plastic", 1.0),
    ("crate", 1.2),
    ("toolbox", 1.2),
    ("float", 1.0),
    ("", 2.5),
)
NOT_SOLID = ("fiducial", "led", "stab", "berm", "rope")


def build_course(stage, tex, *, seed=0, root="/World"):
    z0 = T.Z0
    M.build_seabed(
        stage,
        f"{root}/course_seabed",
        z0=z0,
        texture_dir=tex,
        seed=seed,
        flat_radius=FLAT_RADIUS,
        half=24.0,
    )
    M.scatter_scenery(
        stage,
        f"{root}/scenery",
        seabed_z=z0,
        seed=seed,
        half=22.0,
        keep_out=[(0.0, 0.0, FLAT_RADIUS - 0.5)],
        flat_radius=FLAT_RADIUS,
    )

    v = [np.asarray(p, dtype=np.float64) for p in SERVICE_PIPE]
    for k in range(len(v) - 1):
        P.build_pipe_segment(
            stage,
            f"{root}/course/pipe_{k}",
            v[k],
            v[k + 1],
            z0=z0,
            anode=k % 2 == 0,
            tdir=tex,
            seed=k,
        )

    (px, py), pyaw = T.PANEL
    P.build_inspection_panel(stage, f"{root}/course/panel", (px, py, z0), pyaw, tdir=tex)

    M.build_dock_station(
        stage, f"{root}/course/dock", bay_center=T.DOCK_BAY, seabed_z=z0, texture_dir=tex, seed=seed
    )
    make_solid(stage, f"{root}/course")
    return [(*p, z0 + PIPE_ALTITUDE) for p in v]


def dress_props(stage, tex, prop_root="/World/props"):
    from pxr import Usd, UsdGeom

    from isaacswim_bench.tasks.sample import COLORS, FLOOR_T
    from isaacswim_bench.tasks.valve import MANIFOLD

    def body(name):
        prim = stage.GetPrimAtPath(f"{prop_root}/{name}")
        return prim if prim.IsValid() else None

    def hide(name):
        prim = body(name)
        if prim is not None:
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(p).MakeInvisible()
        return prim

    n = 0
    for k, c in enumerate(COLORS):
        if hide(f"sample_sample_{c}") is not None:
            P.build_sample(
                stage, f"{prop_root}/sample_sample_{c}/look", c, (0, 0, 0), 0.0, seed=k, tdir=tex
            )
            n += 1
    if hide("sample_basket_floor") is not None:
        P.build_sample_crate(
            stage, f"{prop_root}/sample_basket_floor/look", (0, 0, -FLOOR_T / 2), 0.0, tdir=tex
        )
        for k in range(4):
            hide(f"sample_basket_wall_{k}")
        n += 1
    if hide("valve_manifold") is not None:
        P.build_valve_manifold(
            stage, f"{prop_root}/valve_manifold/look", (0, 0, -MANIFOLD[2] / 2), 0.0, tdir=tex
        )
        hide("valve_stem")
        hide("valve_valve_plate")
        n += 1
    if hide("valve_handle") is not None:
        P.build_tbar(stage, f"{prop_root}/valve_handle/look", (0, 0, 0), 0.0, tdir=tex)
        n += 1
    return n


def make_solid(stage, root):
    from pxr import Usd, UsdGeom, UsdPhysics

    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        name = str(prim.GetPath())[len(root) :].lower()
        if any(k in name for k in NOT_SOLID):
            continue
        if prim.IsA(UsdGeom.Mesh):
            if len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or ()) < 8:
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        elif prim.IsA(UsdGeom.Sphere) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
            UsdPhysics.CollisionAPI.Apply(prim)
        else:
            continue
        n += 1
    return n


def light(stage, root="/World"):
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, f"{root}/DomeLight")
    dome.CreateIntensityAttr().Set(350.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.45, 0.65, 0.85))
    sun = UsdLux.DistantLight.Define(stage, f"{root}/sun")
    sun.CreateIntensityAttr(1100.0)
    sun.CreateAngleAttr(4.0)
    sun.CreateColorAttr(Gf.Vec3f(0.75, 0.9, 1.0))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25.0, -15.0, 0.0))


def scene_roots(root="/World", reef=None):
    roots = tuple(p.replace("/World", root, 1) for p in ROOTS)
    return (*roots, f"{root}/Reef") if reef else roots


def dress(
    stage,
    texture_dir,
    *,
    seed=0,
    ground="/World/seabed",
    root="/World",
    origin=(0.0, 0.0, 0.0),
    reef=None,
):
    from pxr import UsdGeom

    if reef and root != "/World":
        raise ValueError("the reef level is one shared scene; use a single environment")
    prim = stage.GetPrimAtPath(ground)
    if prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
    if root == "/World":
        light(stage, root=root)
    build_course(stage, texture_dir, seed=seed, root=root)
    dress_props(stage, texture_dir, prop_root=f"{root}/props")
    if reef:
        from .reef import apply_reef

        apply_reef(stage, reef)
    roots = scene_roots(root, reef)
    if any(origin):
        from pxr import Gf

        for path in roots:
            prim = stage.GetPrimAtPath(path)
            if not prim.GetTypeName():
                prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            UsdGeom.Xformable(prim).AddTranslateOp(opSuffix="environment").Set(Gf.Vec3d(*origin))
    return roots


def label_sonar(osim, stage, roots=ROOTS):
    from pxr import Usd, UsdGeom

    n = 0
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Gprim) or prim.IsInstance():
                path = str(prim.GetPath()).lower()
                r = next(v for k, v in REFLECTIVITY if k in path)
                osim.set_sonar_reflectivity(str(prim.GetPath()), r)
                n += 1
    return n


__all__ = [
    "build_course",
    "dress",
    "dress_props",
    "label_sonar",
    "make_solid",
    "light",
    "ROOTS",
    "scene_roots",
    "FLAT_RADIUS",
]

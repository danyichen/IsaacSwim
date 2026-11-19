from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from ..service import SERVICE_PIPE

REEF_PRIM = "/World/Reef"
FRAGMENT_LIFT = 0.0019
HOLDER_DROP = -0.005
SUN = (1800.0, 5.0, (1.0, 0.97, 0.90))
DOME = (450.0, (0.45, 0.65, 0.85))
PANEL_PARTS = ("gauge", "gauge_rim", "hotstab", "cp_point", "bolt20_130")
MANIFOLD_PARTS = ("pipe_in-1", "riser-1", "riser1")
DOCK_KEPT = ("fiducial", "mat")


def _gprims(prim):
    return (p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Gprim))


def _hide(stage, path, *, solid=True):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    for item in _gprims(prim):
        UsdGeom.Imageable(item).MakeInvisible()
    if not solid:
        _disable_collision(prim)


def _disable_collision(prim):
    for item in Usd.PrimRange(prim):
        if item.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(item).GetCollisionEnabledAttr().Set(False)


def _replace_look(stage, body, name, asset, lift):
    prim = stage.GetPrimAtPath(body)
    if not prim:
        return
    old = stage.GetPrimAtPath(f"{body}/look")
    if old:
        UsdGeom.Imageable(old).MakeInvisible()
    look = UsdGeom.Xform.Define(stage, f"{body}/{name}")
    look.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, lift))
    look.GetPrim().GetReferences().SetReferences([Sdf.Reference(str(asset))])


def _hide_course(stage):
    for path in ("/World/course_seabed", "/World/scenery"):
        _hide(stage, path)
    dock = stage.GetPrimAtPath("/World/course/dock")
    if dock:
        for child in dock.GetChildren():
            name = child.GetName()
            if name not in DOCK_KEPT and "led" not in name:
                _hide(stage, str(child.GetPath()))
    for i in range(len(SERVICE_PIPE) - 1):
        _hide(stage, f"/World/course/pipe_{i}", solid=False)
    for part in PANEL_PARTS:
        _hide(stage, f"/World/course/panel/{part}", solid=False)
    for part in MANIFOLD_PARTS:
        _hide(stage, f"/World/props/valve_manifold/look/{part}")


def _light(stage):
    intensity, angle, color = SUN
    sun = UsdLux.DistantLight.Get(stage, "/World/sun")
    sun.GetIntensityAttr().Set(intensity)
    sun.GetAngleAttr().Set(angle)
    sun.GetColorAttr().Set(Gf.Vec3f(*color))
    intensity, color = DOME
    dome = UsdLux.DomeLight.Get(stage, "/World/DomeLight")
    dome.GetIntensityAttr().Set(intensity)
    dome.GetColorAttr().Set(Gf.Vec3f(*color))


def apply_reef(stage, folder):
    root = Path(folder)
    level = root / "reef.usdc"
    if not level.is_file():
        raise FileNotFoundError(f"the reef level is missing: {level}")
    _hide_course(stage)
    for color in ("red", "yellow", "green"):
        _replace_look(stage, f"/World/props/sample_sample_{color}", "CoralFragment",
                      root / f"coral_fragment_{color}.usdc", FRAGMENT_LIFT)
    _replace_look(stage, "/World/props/sample_basket_floor", "NurseryHolder",
                  root / "nursery_holder.usdc", HOLDER_DROP)
    reef = UsdGeom.Xform.Define(stage, REEF_PRIM)
    reef.GetPrim().GetReferences().SetReferences([Sdf.Reference(str(level))])
    ground = stage.GetPrimAtPath("/World/seabed")
    if ground:
        _disable_collision(ground)
    _light(stage)
    return reef

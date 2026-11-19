from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from isaacswim.assets import cache_dir
from isaacswim.assets.bluerov.build_urdf import PinnedSource, fetch_pinned

CACHE = cache_dir()
SRC = CACHE / "bluerov2_heavy_alpha"
OUT = CACHE / "bluerov2_heavy_robotiq"

_G = "grippers/Robotiq_2F_85/"
ROBOTIQ = PinnedSource(
    key="robotiq-isaacsim-assets",
    repo="robotiq/isaacsim_assets",
    sha="646e8a5dc0ff2a52ff8fa4bb4310c89484390129",
    files={
        "LICENSE": "f162a2346c734d4baaae9d7fa554317654b2af42c87b757af3c01e007e85758e",
        "README.md": "09d1b0a89f3403751f4818e6afcafc426ed47d0bcf66c56f03b21090b3cab3e6",
        _G + "PACKAGE-LICENSES/LICENSE":
            "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
        _G + "Robotiq_2F_85.usda":
            "483b4b8e0dd237f9a5eaf92578f1b9b2d8a8e51bd65a5c662500bd6d780acd43",
        _G + "configuration/Robotiq_2F_85_config_physics_compliant.usda":
            "c7ea028ed9f7bd876f7f96f51ece9840dd0de1683a35271180c9520091d09c0e",
        _G + "configuration/Robotiq_2F_85_config_physics_parallel_grip.usda":
            "f3586de59895cb93081890e1e68caf142121b3d101f0714c983b2560836ad7a9",
        _G + "configuration/Robotiq_2F_85_robot.usda":
            "f60288330aedcd1682dcef7b9d50a5de914a7be818039a26b5da956effe2e3b6",
        _G + "materials/LICENSE":
            "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
        _G + "materials/materials.usd":
            "48eca813596e78486469edc70f825c9bb9913836f63e4007559920820e8e9d9d",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_Finger1step_JFT.usd":
            "1d8f971e8d9e6b140ae406adbe4ebf17831c8c298ce3ce07577b7fac3590bb8c",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_basestep_JFX.usd":
            "a9082107ce6d92b69945ca81f22b271c24c5fba338655e6b838c517d81039826",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_finger2step_JFP.usd":
            "82d1355f767006bdcb1376e8fd495c222120b18776d6d9cccdb199c0ef3d240d",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_finger3step_JFL.usd":
            "2daaae8c788e7e74fa54a5850854b4dcd2ee6d739a2873db63c1fc5db157e87c",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_finger4step_JFH.usd":
            "8d7e3fce9d1c70ab65a676f3efdfa5fd1d161a66793a36bed408509e33b64fe5",
        _G + "parts/Defeatured_2F_85_PAD_OPEN_fingertipsstep_JFD.usd":
            "700409224e3d2c4e0a0e2ba4b4da4f644babd74207d660c034ce53b255b3e876",
        _G + "parts/Robotiq_2F_85_dowel_pin.usda":
            "87bfc3752e993f8c89bc8a4218dda93918f6dca793b5c6d6441a4bbe3d904ce5",
        _G + "parts/TSF_85_case.usd":
            "3465da112b4cd8098b0d2dc29d861db8a4efe0530cc4ad3933e52b3ca60c12fd",
        _G + "payloads/Robotiq_2F_85_body_mass_physics.usda":
            "21e46529d377d8e300d4a4e6bd3e09d9c40d1b905e3a3dfce27fb982453699db",
        _G + "payloads/Robotiq_2F_85_compliant_physics.usda":
            "a45eaf1081737cc8588537399cb535889f613e4299547efb51531bc3674b89a9",
        _G + "payloads/Robotiq_2F_85_fingertip_physics.usda":
            "ac7e592beb00e3263a9bae376a8c3cd49daa98986743c39c4bb02c004b8b475d",
        _G + "payloads/Robotiq_2F_85_fingertip_tsf85.usda":
            "aabc10f07be808c8abae93f1a8f34c3c218937916b433e7e2cc4afa4004f0789",
        _G + "payloads/Robotiq_2F_85_kinematics_physics.usda":
            "0ccc798f315e6e2b4baa7c11abfa9f58906d0c0462d3b404ea35560e135bb8a1",
        _G + "payloads/Robotiq_2F_85_newton_common_physics.usda":
            "6952bd5be763fcd121d45be062e54856eaac4b870a3963f0a0aa22d3ebb3258f",
        _G + "payloads/Robotiq_2F_85_newton_compliant_physics.usda":
            "14f68a48202ee8138f9d13dcdd268812995ea594a242a2dffcb807f1a652d59c",
        _G + "payloads/Robotiq_2F_85_newton_fourbar_physics.usda":
            "2d116a900db6ceb4d3fb9d48a208d53a6cd41bc7189896027f74689a30285137",
        _G + "payloads/Robotiq_2F_85_newton_parallel_grip_physics.usda":
            "4d9f0039d5c76faf16bf46ac8241cd526b5c7be40514e108e47fa8ce6b527145",
        _G + "payloads/Robotiq_2F_85_parallel_grip_dowel_pins.usda":
            "c4df764c154ef6086122d36694dfa4956c2f78b73ea4cf2f9749127a24a95854",
        _G + "payloads/Robotiq_2F_85_parallel_grip_physics.usda":
            "0f7e98ea8d31f584121a9cc211351b3042ce451dcf898c383d470fe05c479b99",
        _G + "payloads/Robotiq_2F_85_physx_common_physics.usda":
            "034630ddb2f47d1f1f65fa22c37031aadfa5395810754a211f19b80851566bd0",
        _G + "payloads/Robotiq_2F_85_tactile_effective_response.usda":
            "1f2a2bdb87876531b2bc00411957c95e6563d33555ff0c8378875b4e77ea6eef",
        _G + "payloads/Robotiq_2F_85_tactile_raycast.usda":
            "f21bab8e7ea3edb4f9bd0787d7620cdc5862c3c60064a5febafe8a3e52129335",
        _G + "payloads/Robotiq_2F_85_tactile_zones.usda":
            "424a6d5442762370e6a264f2522e345b398214bb680a5c3b3606bbd4ceeef54e",
        _G + "payloads/base.usda":
            "aad894840730a45edd7815b8ca8699cb9252791683af8e98d28fb2e604db2d71",
        _G + "payloads/geometries.usd":
            "43f1718720ea76cf8c69d5031de8d215806c79a54618417c431c0f17fe0758f3",
        _G + "payloads/instances.usda":
            "f3354daa0ef1d865ac397413b732636e56d9affc5ce48240a174b30bea5309f6",
        _G + "payloads/materials.usda":
            "a5b6d1edfd4eb76df8a64f4ed694275950c5ec1d5ba56beabefbeb1cec562bb4",
    },
    notices=("LICENSE", _G + "PACKAGE-LICENSES/LICENSE", _G + "materials/LICENSE"),
    staged=("", ""),
)
ROBOTIQ_USD = _G + "Robotiq_2F_85.usda"
VARIANTS = {"Physics": "Physx_parallel_grip", "Fingertip": "Standard"}

ROBOTIQ_SOURCE = """\
Robotiq 2F-85 Isaac Sim asset, fetched and modified by IsaacSwim.

Source: {url}/{asset}
Licenses: Robotiq-authored content is BSD-3-Clause (ROBOTIQ_LICENSE, which also
lists the third-party attributions).  Portions were adapted by Robotiq from
NVIDIA Corporation's Robotiq 2F-85 asset and remain under the Creative Commons
Attribution 4.0 International License (ROBOTIQ_CC-BY-4.0.txt,
https://creativecommons.org/licenses/by/4.0/).
Attribution: original work by NVIDIA Corporation; modifications Copyright (c)
2026, ROBOTIQ; further modifications by IsaacSwim, listed below.
IsaacSwim's changes: selected the {variants} variants and flattened the asset;
renamed base_link to robotiq_base_link; removed its articulation root and its
world root_joint; mounted it on the Reach Alpha 5 in place of the standard
jaws.  These files are not covered by IsaacSwim's source license, and neither
Robotiq nor NVIDIA endorses IsaacSwim.
"""

MOUNT_LINK = "alpha_m1_link"
MOUNT_Z = 0.09975
JAW_PRIMS = ("alpha_standard_jaws_rs1_130_link", "alpha_standard_jaws_rs1_139_link",
             "joints/alpha_standard_jaws_rs1_130_joint",
             "joints/alpha_standard_jaws_rs1_139_joint")
JAW_LINKS = ("alpha_standard_jaws_base_link", "alpha_standard_jaws_rs1_130_link",
             "alpha_standard_jaws_rs1_139_link")
ALUMINIUM = 2700.0


def fetch(*, force: bool = False, cache: Path | None = None) -> Path:
    return fetch_pinned(ROBOTIQ, force=force, cache=cache) / ROBOTIQ_USD


def retain_robotiq_notices(root: Path, out: Path) -> None:
    shutil.copy2(root / "LICENSE", out / "ROBOTIQ_LICENSE")
    shutil.copy2(root / _G / "PACKAGE-LICENSES/LICENSE", out / "ROBOTIQ_CC-BY-4.0.txt")
    variants = ", ".join(f"{k}={v}" for k, v in VARIANTS.items())
    (out / "ROBOTIQ_ASSET_SOURCE.txt").write_text(
        ROBOTIQ_SOURCE.format(url=ROBOTIQ.url, asset=ROBOTIQ_USD, variants=variants),
        encoding="utf-8")


def _remap(layer, old, new) -> int:
    from pxr import Sdf

    n = 0

    def fix(p):
        return p.ReplacePrefix(old, new) if p.HasPrefix(old) else p

    def visit(path):
        nonlocal n
        spec = layer.GetObjectAtPath(path)
        for attr in ("targetPathList", "connectionPathList"):
            lst = getattr(spec, attr, None)
            if lst is None:
                continue
            for op in ("explicitItems", "prependedItems", "appendedItems"):
                items = list(getattr(lst, op))
                new_items = [fix(p) for p in items]
                if new_items != items:
                    setattr(lst, op, new_items)
                    n += 1
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return n


def flatten_robotiq(src: Path, out: Path) -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(src), Usd.Stage.LoadAll)
    top = stage.GetDefaultPrim()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, value in VARIANTS.items():
            if not top.GetVariantSets().GetVariantSet(name).SetVariantSelection(value):
                raise RuntimeError(f"gripper asset has no {name}={value} variant")
    flat = stage.Flatten()

    root = Sdf.Path("/Robotiq_2F_85/Robotiq_2F_85")
    old, new = root.AppendChild("base_link"), root.AppendChild("robotiq_base_link")
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(old, new)
    if not flat.Apply(edit):
        raise RuntimeError("could not rename the gripper's base_link")
    fixed = _remap(flat, old, new)

    removed = []
    for prim in list(_specs(flat, root)):
        if prim.name == "root_joint":
            removed.append(str(prim.path))
            del prim.nameParent.nameChildren[prim.name]
    if not removed:
        raise RuntimeError("gripper asset has no root_joint to remove")

    inner = flat.GetPrimAtPath(root)
    op = inner.GetInfo("apiSchemas")
    kept = [s for s in (op.GetAddedOrExplicitItems() if op else [])
            if "Articulation" not in s]
    inner.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(kept or ["IsaacRobotAPI"]))
    for key in list(inner.properties.keys()):
        if key.startswith(("physxArticulation:", "physics:articulation")):
            inner.RemoveProperty(inner.properties[key])
    if flat.GetPrimAtPath("/Render"):
        del flat.rootPrims["Render"]
    flat.Export(str(out))
    print(f"  {out.name}: {VARIANTS}, base_link -> robotiq_base_link "
          f"({fixed} paths remapped), articulation root and {len(removed)} world joint removed")


def _specs(layer, root):
    stack = [layer.GetPrimAtPath(root)]
    while stack:
        spec = stack.pop()
        yield spec
        stack.extend(spec.nameChildren)


def gripper_hydro(flat_usd: Path) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(flat_usd))
    root = st.GetPrimAtPath("/Robotiq_2F_85/Robotiq_2F_85")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    out = {}
    for link in root.GetChildren():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        own = (UsdPhysics.MassAPI(link).GetMassAttr().Get()
               if link.HasAPI(UsdPhysics.MassAPI) else None)
        mass = float(own) if own else 0.0
        if not own:
            for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                m = (UsdPhysics.MassAPI(p).GetMassAttr().Get()
                     if p.HasAPI(UsdPhysics.MassAPI) else None)
                if m:
                    mass += float(m)
        box = cache.ComputeRelativeBound(link, link).ComputeAlignedRange()
        lo, hi = box.GetMin(), box.GetMax()
        c = (lo + hi) / 2
        half = (hi - lo) / 2
        out[link.GetName()] = {
            "mass": mass, "volume": mass / ALUMINIUM,
            "pos": [float(v) for v in c], "semiaxes": [max(float(v), 0.004) for v in half]}
    return out


def strip_urdf(src: Path, dst: Path) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(src)
    robot = tree.getroot()
    for el in list(robot):
        if el.tag == "link" and el.get("name") in JAW_LINKS:
            robot.remove(el)
        elif el.tag == "joint" and el.find("child").get("link") in JAW_LINKS:
            robot.remove(el)
    ET.SubElement(robot, "link", name="robotiq_tool_link")
    j = ET.SubElement(robot, "joint", name="robotiq_tool_joint", type="fixed")
    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(j, "parent", link="alpha_jaws_base_link")
    ET.SubElement(j, "child", link="robotiq_tool_link")
    rel = Path("..") / src.parent.name
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not re.match(r"^(\w+:|/)", fn):
            mesh.set("filename", (rel / fn).as_posix())
    tree.write(dst, xml_declaration=True, encoding="utf-8")


def compose(vehicle_usd: Path, gripper_usd: Path, out_usd: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    veh = Usd.Stage.Open(str(vehicle_usd))
    top = veh.GetDefaultPrim().GetPath()
    by_name = {p.GetName(): p.GetPath() for p in Usd.PrimRange(veh.GetPrimAtPath(top))}
    m1 = veh.GetPrimAtPath(by_name[MOUNT_LINK]) if MOUNT_LINK in by_name else None
    if not m1:
        raise RuntimeError(f"no {MOUNT_LINK} in {vehicle_usd}")
    mount_w = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, MOUNT_Z)) * \
        UsdGeom.Xformable(m1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    top_w = UsdGeom.Xformable(veh.GetPrimAtPath(top)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())

    st = Usd.Stage.CreateNew(str(out_usd))
    st.GetRootLayer().subLayerPaths.append(
        Path("..", vehicle_usd.parent.name, vehicle_usd.name).as_posix())
    st.SetDefaultPrim(st.GetPrimAtPath(top) or st.OverridePrim(top))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)

    for rel in JAW_PRIMS:
        name = rel.rsplit("/", 1)[-1]
        st.OverridePrim(by_name.get(name, top.AppendPath(rel))).SetActive(False)

    grip = st.DefinePrim(top.AppendChild("robotiq"), "Xform")
    grip.GetReferences().AddReference(gripper_usd.name, "/Robotiq_2F_85")
    UsdGeom.Xformable(grip).AddTransformOp().Set(mount_w * top_w.GetInverse())

    base = top.AppendPath("robotiq/Robotiq_2F_85/robotiq_base_link")
    joint = UsdPhysics.FixedJoint.Define(st, top.AppendPath("joints/robotiq_mount_joint"))
    joint.CreateBody0Rel().SetTargets([m1.GetPath()])
    joint.CreateBody1Rel().SetTargets([base])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, MOUNT_Z))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    st.GetRootLayer().Save()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Attach a Robotiq 2F-85 to the Alpha 5.")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--refetch", action="store_true", help="re-download the pinned asset")
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    if out.resolve() == src.resolve():
        raise SystemExit("--out must differ from --src")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("fetching the 2F-85 asset ...")
    edit = fetch(force=args.refetch)
    flat = out / "robotiq_2f85.usd"
    flatten_robotiq(edit, flat)
    compose(src / f"{src.name}.usd", flat, out / f"{out.name}.usda")
    strip_urdf(src / f"{src.name}.urdf", out / f"{out.name}.urdf")
    if not (src / "REACH_LICENSE").is_file():
        raise SystemExit(f"{src} has no REACH_LICENSE -- rebuild it with build_urdf")
    for f in ("thrusters.json", "authority.json", "UPSTREAM_LICENSE", "UPSTREAM_README.md",
              "REACH_LICENSE", "REACH_NOTICE.txt", "ASSET_SOURCES.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    if (src / "SOURCE_NOTICES").is_dir():
        shutil.copytree(src / "SOURCE_NOTICES", out / "SOURCE_NOTICES", dirs_exist_ok=True)
    (out / "NVIDIA_ASSET_SOURCE.txt").unlink(missing_ok=True)
    retain_robotiq_notices(edit.parents[2], out)
    hydro = gripper_hydro(flat)
    (out / "robotiq_hydro.json").write_text(json.dumps(hydro, indent=2))
    total = sum(v["mass"] for v in hydro.values())
    print(f"  gripper: {len(hydro)} links, {total:.3f} kg, "
          f"{sum(v['volume'] for v in hydro.values()) * 1e3:.3f} L displaced")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

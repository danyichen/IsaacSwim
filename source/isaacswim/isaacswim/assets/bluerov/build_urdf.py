#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from isaacswim.assets import cache_dir

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PinnedSource:
    key: str
    repo: str
    sha: str
    files: dict[str, str]
    notices: tuple[str, ...]
    staged: tuple[str, str]

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}/tree/{self.sha}"

    def raw(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.sha}/{path}"

    def lfs(self, path: str) -> str:
        return f"https://media.githubusercontent.com/media/{self.repo}/{self.sha}/{path}"


_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


UVMS = PinnedSource(
    key="uvms-simulator",
    repo="edxmorgan/uvms-simulator",
    sha="98e8ba1e5df5a2c87a0fc78898701618c53ca1fd",
    files={
        "LICENSE": "8486a10c4393cee1c25392769ddd3b2d6c242d6ec7928e1414efff7dfb2f07ef",
        "README.md": "28b62d61b3483d38814f24e63f00acf407163dc74c5445005af98e0ff6a59252",
        "description/blue/xacro/bluerov2_heavy/urdf.xacro":
            "dcf25fd8ae65304ec7d01477ad5a1956467b038cbec5d14abed93cdb2b036e94",
        "description/blue/xacro/camera/urdf.xacro":
            "849413b3ec037a39108f00823f2e938827a7efc891f61ac2676260a74dd9be39",
        "description/blue/xacro/t200/urdf.xacro":
            "472b187e26005354046118ffc6c053884425893d5cdcccdb8c9fe998e5d9073c",
        "description/blue/meshes/bluerov2_heavy/bluerov2_heavy.dae":
            "c34f57e1a67d5924145377f5086358e307de8de9bfbd78beeac9468ac1395df5",
        "description/blue/meshes/t200/ccw_prop.dae":
            "e1918897f91bfa50720cec6feb3053315560878eaa5df8cc9c7519c4484f01dd",
        "description/blue/meshes/t200/cw_prop.dae":
            "9ac826b9c380759777869414fc79d19d880a7c66473886874010aa24101f7a05",
    },
    notices=("LICENSE", "README.md"),
    staged=("description/", ""),
)

REACH = PinnedSource(
    key="reach-alpha",
    repo="Robotic-Decision-Making-Lab/alpha",
    sha="5e1a6648b78584d207e79890bc3905b770892dcd",
    files={
        "LICENSE": "c740b074d5e7a563cf96032b4f8b9fd220fa110eee264e3f47998e7c10101341",
        "README.md": "9784b03e18b70ce68c40c601a2fe13bfb39bbe3507e24d6036985a9f55d0c9a8",
        "alpha_description/LICENSE":
            "e4da13b8ee8adf3c184f3cfaa765c1b002efd9e1b7a200fe7c507a30391f8fb9",
        "alpha_description/meshes/LICENSE":
            "ee0b23a2669a68b89e3d08a64ff4901b1d832b40cdd535dd2de4a7db0aae8546",
        "alpha_description/xacro/LICENSE":
            "ee0b23a2669a68b89e3d08a64ff4901b1d832b40cdd535dd2de4a7db0aae8546",
        "alpha_description/xacro/alpha.urdf.xacro":
            "7e00261ad52e083d2871309a342c7b5047b0b565b4f616d7b1df1c2b0879e290",
        "alpha_description/xacro/end_effectors/standard_jaws.urdf.xacro":
            "44df94ef6b6b6d2ad624663f4a74dfb07e0e76aa4c84f6c12f2c8624797c3cd9",
        "alpha_description/meshes/M2-1-1.stl":
            "55d98e61e7522c07d83c41b8aa878bf5c14d7bbd8358d7b7db6e8284b3490ada",
        "alpha_description/meshes/M2-1-3.stl":
            "0ea958b96fe16b55c19a1533e32e17d3114f92b3fe3a88cfdb761ded6d32aab2",
        "alpha_description/meshes/M2.stl":
            "a8f5cd383ff82a42d66cac12bf8cdaae84a960ae308700d0eb1815f0e925d979",
        "alpha_description/meshes/M3-INLINE.stl":
            "9841a0b3dcbc0bdc1fbdc45d4643ca9b77ccc1fe235eb2326d95913f49053868",
        "alpha_description/meshes/RS1-100-101-123.stl":
            "205fd28ad9f59d9c13e6f56718119870860450fcb2e0636c3ce9b3fec006a50d",
        "alpha_description/meshes/end_effectors/RS1-124.stl":
            "7981b49ab9c42e6211b547d6963c6a93957b81870e69e7f03f9f7c2567067ab4",
        "alpha_description/meshes/end_effectors/RS1-130.stl":
            "0566fa5e2443b2d5f011dfbf1adfe29910a2041de7d61c3bc8860acc32a164a6",
        "alpha_description/meshes/end_effectors/RS1-139.stl":
            "2db4c9b6ee9d25fbff9c8fc5cc3601e33375a5cae378901739751b763e22cbfc",
    },
    notices=("LICENSE", "README.md", "alpha_description/LICENSE",
             "alpha_description/meshes/LICENSE", "alpha_description/xacro/LICENSE"),
    staged=("alpha_description/", "alpha/"),
)
REACH_LICENSE = "alpha_description/meshes/LICENSE"
SOURCES = (UVMS, REACH)

FIND_SUBSTITUTIONS = {
    "$(find isaacswim_vehicle)": "",
    "$(find ros2_control_blue_reach_5)": "",
    "package://ros2_control_blue_reach_5": "",
    "$(find ${description_package})": "alpha",
}

REACH_NOTICE = """\
Reach Robotics Alpha 5 model -- notice for this generated asset

The Alpha 5 description and meshes used here were downloaded by IsaacSwim's
builder from {url}/alpha_description,
a source that carries Reach Robotics' license.  That license is kept verbatim
in REACH_LICENSE, beside the converted meshes and under
SOURCE_NOTICES/reach-alpha/.  IsaacSwim converted the files for simulation
(URDF, OBJ, USD).

Reach Robotics Pty Ltd gave the IsaacSwim author written authorization to
proceed with simulation use on these conditions, and may revoke it:

  * Download the model only from a source that contains Reach's license file.
  * Do not redistribute the original or the converted files.  Each user
    generates them locally with the builder.
  * Keep Reach's notice and copyright lines with the files.
  * IsaacSwim is not affiliated with, endorsed by or sponsored by Reach
    Robotics.
  * Simulation only: the simulated Alpha 5's behaviour and performance are
    not representative of the real product's performance.

For any other use, contact Reach Robotics.
"""

CACHE = cache_dir()

OUT = CACHE / "bluerov2_heavy_alpha"

ALPHA_JOINT_LIMITS = {
    "alpha_axis_e": (0.0, 5.7),
    "alpha_axis_d": (1.5, 3.4),
    "alpha_axis_c": (0.2, 3.4),
    "alpha_axis_b": (0.0, 5.7),
}

ALPHA_RENAMES = {
    "alpha_base_joint": "alpha_joint",
    "alpha_ee_joint": "alpha_jaws_joint",
    "alpha_ee_base_link": "alpha_jaws_base_link",
    "alpha_standard_jaws_base_joint": "alpha_standard_jaws_joint",
}

FALLBACK_RADIUS = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_tree(root: Path, source: PinnedSource) -> str | None:
    for rel, digest in source.files.items():
        path = root / rel
        if not path.is_file():
            return f"missing {rel}"
        if _sha256(path) != digest:
            return f"{rel} does not match {source.repo}@{source.sha[:8]}"
    for rel in source.notices:
        if not (root / rel).read_bytes().strip():
            return f"{rel} is empty"
    return None


def fetch_pinned(source: PinnedSource, *, force: bool = False,
                 cache: Path | None = None) -> Path:
    dest = (cache or CACHE) / source.key
    stamp = dest / ".pinned"
    if (not force and stamp.is_file() and stamp.read_text().strip() == source.sha
            and _check_tree(dest, source) is None):
        return dest

    print(f"fetching {len(source.files)} files from {source.repo}@{source.sha[:8]}",
          file=sys.stderr)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        for rel, digest in source.files.items():
            with urllib.request.urlopen(source.raw(rel)) as resp:
                data = resp.read()
            if data.startswith(_LFS_POINTER):
                with urllib.request.urlopen(source.lfs(rel)) as resp:
                    data = resp.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"{source.repo}@{source.sha[:8]}: {rel} does not match "
                                   "its pinned SHA-256")
            path = partial / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        problem = _check_tree(partial, source)
        if problem:
            raise RuntimeError(f"{source.repo}@{source.sha[:8]}: {problem}")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    stamp.write_text(source.sha + "\n")
    return dest


def source_input(source: PinnedSource, directory: Path | None = None, *,
                 force: bool = False) -> tuple[Path, str]:
    if directory is None:
        return fetch_pinned(source, force=force), source.url
    if force:
        raise ValueError("a local source directory and --refetch cannot be combined")
    root = Path(directory).expanduser().resolve()
    problem = _check_tree(root, source)
    if problem:
        raise ValueError(f"local {source.key} checkout {root}: {problem}")
    return root, f"local checkout of {source.url}: {root.as_posix()}"


def validate_output_path(inputs: list[Path], out: Path) -> None:
    dest = out.expanduser().resolve()
    for root in inputs:
        root = root.resolve()
        if dest.is_relative_to(root) or root.is_relative_to(dest):
            raise ValueError(f"build output overlaps the source checkout {root}: {dest}")


def retain_notices(roots: dict[str, Path], out: Path) -> None:
    for source in SOURCES:
        root = roots[source.key]
        for rel in source.notices:
            dest = out / "SOURCE_NOTICES" / source.key / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dest)
    shutil.copy2(roots[UVMS.key] / "LICENSE", out / "UPSTREAM_LICENSE")
    shutil.copy2(roots[UVMS.key] / "README.md", out / "UPSTREAM_README.md")
    shutil.copy2(roots[REACH.key] / REACH_LICENSE, out / "REACH_LICENSE")
    (out / "REACH_NOTICE.txt").write_text(REACH_NOTICE.format(url=REACH.url),
                                          encoding="utf-8")


def stage_description(roots: dict[str, Path], workdir: Path) -> Path:
    staged = workdir / "description"
    for source in SOURCES:
        prefix, into = source.staged
        for rel in source.files:
            if rel.startswith(prefix):
                dest = staged / into / rel[len(prefix):]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(roots[source.key] / rel, dest)
    shutil.copy(HERE / "robot.xacro", staged / "robot.xacro")

    for path in staged.rglob("*.xacro"):
        text = original = path.read_text(encoding="utf-8")
        for needle, sub in FIND_SUBSTITUTIONS.items():
            text = text.replace(needle, str(staged / sub) if sub else str(staged))
        if text != original:
            path.write_text(text, encoding="utf-8")
    return staged / "robot.xacro"


def run_xacro(top: Path) -> str:
    try:
        import xacro
    except ImportError:
        raise SystemExit("xacro not importable -- `pip install xacro`") from None

    doc = xacro.process_file(str(top))
    return doc.toxml()


def _inertia_of_collision(link: ET.Element, mass: float) -> tuple[float, float, float] | None:
    col = link.find("collision")
    if col is None:
        return None
    geom = col.find("geometry")
    if geom is None:
        return None

    box = geom.find("box")
    if box is not None:
        x, y, z = (float(v) for v in box.get("size", "0 0 0").split())
        k = mass / 12.0
        return (k * (y * y + z * z), k * (x * x + z * z), k * (x * x + y * y))

    cyl = geom.find("cylinder")
    if cyl is not None:
        r = float(cyl.get("radius", 0.0))
        h = float(cyl.get("length", 0.0))
        radial = mass * (3.0 * r * r + h * h) / 12.0
        return (radial, radial, 0.5 * mass * r * r)

    sph = geom.find("sphere")
    if sph is not None:
        r = float(sph.get("radius", 0.0))
        i = 0.4 * mass * r * r
        return (i, i, i)

    return None


def add_missing_inertials(robot: ET.Element, mass: float) -> list[str]:
    fixed = []
    for link in robot.findall("link"):
        if link.find("inertial") is not None:
            continue
        name = link.get("name", "?")
        diag = _inertia_of_collision(link, mass)
        if diag is None:
            i = 0.4 * mass * FALLBACK_RADIUS**2
            diag = (i, i, i)

        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{mass:.6g}"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{diag[0]:.6g}", "ixy": "0", "ixz": "0",
                "iyy": f"{diag[1]:.6g}", "iyz": "0",
                "izz": f"{diag[2]:.6g}",
            },
        )
        link.remove(inertial)
        link.insert(0, inertial)
        fixed.append(name)
    return fixed


def _subtree_links(robot: ET.Element, root: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in robot.findall("joint"):
        p = joint.find("parent")
        c = joint.find("child")
        if p is not None and c is not None:
            children.setdefault(p.get("link", ""), []).append(c.get("link", ""))

    seen, stack = {root}, [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _drop_links(robot: ET.Element, names: set[str]) -> None:
    for link in [x for x in robot.findall("link") if x.get("name") in names]:
        robot.remove(link)
    for joint in list(robot.findall("joint")):
        p, c = joint.find("parent"), joint.find("child")
        if (p is not None and p.get("link") in names) or (c is not None and c.get("link") in names):
            robot.remove(joint)


def rework_jaws(robot: ET.Element, mode: str) -> list[str]:
    log: list[str] = []
    if mode == "mimic":
        return ["jaws left as upstream's mimic chain (Isaac may not import it)"]

    if mode == "none":
        base = next(
            (x.get("name") for x in robot.findall("link")
             if (x.get("name") or "").endswith("jaws_base_link")),
            None,
        )
        if base:
            doomed = _subtree_links(robot, base)
            _drop_links(robot, doomed)
            log.append(f"dropped {len(doomed)} gripper links below {base}")
        return log

    for link in list(robot.findall("link")):
        if (link.get("name") or "").endswith("push_rod"):
            _drop_links(robot, {link.get("name") or ""})
            log.append(f"removed {link.get('name')} and its prismatic joint")

    for joint in robot.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
            log.append(f"{joint.get('name')}: mimic dropped, now independently actuated")
    return log


_UNSAFE_NAME = re.compile(rb'(<\w+\b[^>]*?\bname=")([^"]*)(")')


def sanitize_collada_names(path: Path) -> int:
    raw = path.read_bytes()
    count = 0

    def fix(match: re.Match[bytes]) -> bytes:
        nonlocal count
        name = match.group(2)
        safe = re.sub(rb"[^A-Za-z0-9_]", b"_", name)
        if safe and safe[:1].isdigit():
            safe = b"m_" + safe
        if safe != name:
            count += 1
        return match.group(1) + safe + match.group(3)

    patched = _UNSAFE_NAME.sub(fix, raw)
    if count:
        path.write_bytes(patched)
    return count


def transcode_mesh(path: Path, *, bake_colours: bool) -> Path | None:
    try:
        import numpy as np
        import trimesh
    except ImportError:
        return None

    scene = trimesh.load(str(path), force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        if bake_colours:
            base = getattr(getattr(mesh.visual, "material", None), "baseColorFactor", None)
            colour = np.array(base[:4] if base is not None else [128, 128, 128, 255],
                              dtype=np.uint8)
            if tuple(colour[:3]) == (255, 0, 0):
                colour = np.array([26, 26, 26, 255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.tile(colour, (len(mesh.vertices), 1)),
            )
        mesh.apply_transform(transform)
        parts.append(mesh)

    merged = trimesh.util.concatenate(parts)
    if not np.allclose(scene.extents, merged.extents, atol=1e-4):
        raise SystemExit(
            f"{path.name}: transcode changed the bounding box "
            f"{scene.extents} -> {merged.extents}; refusing to write a mesh whose "
            "orientation no longer matches the URDF"
        )

    obj = path.with_suffix(".obj")
    merged.export(obj)
    path.unlink()
    return obj


def rewrite_meshes(robot: ET.Element, staged: Path, out: Path,
                   mesh_format: str = "obj") -> int:
    mesh_root = out / "meshes"
    copied: dict[str, str] = {}

    for mesh in robot.iter("mesh"):
        src_uri = mesh.get("filename", "")
        src = Path(src_uri.removeprefix("file://"))
        if not src.is_absolute():
            continue
        if src_uri in copied:
            mesh.set("filename", copied[src_uri])
            continue
        if not src.is_file():
            raise SystemExit(f"mesh referenced but missing: {src}")

        rel = src.relative_to(staged)
        flat = Path(rel.parts[0]) / (re.sub(r"[^A-Za-z0-9_]", "_", rel.stem) + rel.suffix)
        dst = mesh_root / flat
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

        suffix = dst.suffix.lower()
        if suffix == ".dae":
            sanitize_collada_names(dst)

        if mesh_format == "obj" and suffix in (".dae", ".stl"):
            converted = transcode_mesh(dst, bake_colours=(suffix == ".dae"))
            if converted is None:
                print(f"  WARNING: {flat.name} left as {suffix[1:].upper()} -- "
                      "`pip install trimesh pycollada` to transcode it; "
                      "Isaac Sim 5.1 cannot import these", file=sys.stderr)
            else:
                print(f"  transcoded {flat.name} -> {converted.name}", file=sys.stderr)
                dst, flat = converted, flat.with_suffix(".obj")

        ref = f"meshes/{flat.as_posix()}"
        copied[src_uri] = ref
        mesh.set("filename", ref)

    return len(copied)


def drop_footprint(robot: ET.Element) -> list[str]:
    if not any(x.get("name") == "base_footprint" for x in robot.findall("link")):
        return []
    _drop_links(robot, {"base_footprint"})
    return ["dropped base_footprint (massless ROS ground-projection frame)"]


def strip_ros_only(robot: ET.Element) -> list[str]:
    dropped = []
    for tag in ("gazebo", "ros2_control", "transmission"):
        for el in robot.findall(tag):
            robot.remove(el)
            dropped.append(tag)
    return dropped


def _rpy_to_matrix(r: float, p: float, y: float) -> list[list[float]]:
    cr, sr, cp, sp, cy, sy = (
        math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y),
    )
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _compose(a_xyz, a_rot, b_xyz, b_rot):
    xyz = [a_xyz[i] + sum(a_rot[i][k] * b_xyz[k] for k in range(3)) for i in range(3)]
    rot = [[sum(a_rot[i][k] * b_rot[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return xyz, rot


def _is_thruster(name: str) -> bool:
    return "thruster" in name.lower()


def weld_thrusters(robot: ET.Element) -> list[str]:
    log = []
    props = {j.find("child").get("link") for j in robot.findall("joint")
             if j.find("child") is not None and _is_thruster(j.find("child").get("link", ""))}

    n = 0
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is None or child.get("link") not in props:
            continue
        if joint.get("type") == "continuous":
            joint.set("type", "fixed")
            for axis in joint.findall("axis"):
                joint.remove(axis)
            n += 1
    if n:
        log.append(f"welded {n} propeller joints (continuous -> fixed)")

    fixed_inertia = 0
    for link in robot.findall("link"):
        if link.get("name") not in props:
            continue
        inertia = link.find("inertial/inertia")
        mass_el = link.find("inertial/mass")
        if inertia is None or mass_el is None:
            continue
        m, r = float(mass_el.get("value", 0.0)), 0.05
        radial, axial = 0.25 * m * r * r, 0.5 * m * r * r
        inertia.set("ixx", f"{radial:.6g}")
        inertia.set("iyy", f"{radial:.6g}")
        inertia.set("izz", f"{axial:.6g}")
        fixed_inertia += 1
    if fixed_inertia:
        log.append(f"replaced placeholder inertia on {fixed_inertia} propeller links")
    return log


def extract_thrusters(robot: ET.Element, base: str = "base_link") -> list[dict]:
    joints = {}
    for j in robot.findall("joint"):
        p, c, o = j.find("parent"), j.find("child"), j.find("origin")
        if p is None or c is None:
            continue
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        joints[c.get("link", "")] = (p.get("link", ""), xyz, _rpy_to_matrix(*rpy), j.get("type"))

    local_axis = (0.0, 0.0, -1.0)

    out = []
    for link in robot.findall("link"):
        name = link.get("name") or ""
        if not _is_thruster(name):
            continue

        xyz, rot, node, rigid = [0.0, 0.0, 0.0], _rpy_to_matrix(0, 0, 0), name, True
        while node in joints and node != base:
            parent, j_xyz, j_rot, j_type = joints[node]
            if j_type != "fixed":
                rigid = False
                break
            xyz, rot = _compose(j_xyz, j_rot, xyz, rot)
            node = parent
        if not rigid or node != base:
            continue

        axis = [sum(rot[i][k] * local_axis[k] for k in range(3)) for i in range(3)]
        visual = link.find("visual/geometry/mesh")
        mesh = (visual.get("filename", "") if visual is not None else "")

        out.append({
            "link": name,
            "position": [round(v, 6) for v in xyz],
            "axis": [round(v, 6) for v in axis],
            "ccw": "ccw_prop" in mesh,
        })
    return sorted(out, key=lambda t: t["link"])


def allocation_matrix(thrusters: list[dict]) -> list[list[float]]:
    cols = []
    for t in thrusters:
        a, p = t["axis"], t["position"]
        tau = [p[1] * a[2] - p[2] * a[1], p[2] * a[0] - p[0] * a[2], p[0] * a[1] - p[1] * a[0]]
        cols.append([*a, *tau])
    return [[round(c[r], 6) for c in cols] for r in range(6)]


def adapt_reach_alpha(robot: ET.Element) -> list[str]:
    log = []
    for el in list(robot.iter("link")) + list(robot.iter("joint")):
        new = ALPHA_RENAMES.get(el.get("name", ""))
        if new:
            el.set("name", new)
    for el in list(robot.iter("parent")) + list(robot.iter("child")):
        new = ALPHA_RENAMES.get(el.get("link", ""))
        if new:
            el.set("link", new)
    for mimic in robot.iter("mimic"):
        new = ALPHA_RENAMES.get(mimic.get("joint", ""))
        if new:
            mimic.set("joint", new)
    log.append(f"renamed {len(ALPHA_RENAMES)} Reach frames to IsaacSwim's alpha_jaws_* names")

    tools = {x.get("name") for x in robot.findall("link")
             if (x.get("name") or "").startswith("alpha_") and x.get("name", "").endswith("_tool")}
    if tools:
        _drop_links(robot, tools)
        log.append(f"dropped tool marker frame(s): {', '.join(sorted(tools))}")

    for joint in robot.findall("joint"):
        lim = ALPHA_JOINT_LIMITS.get(joint.get("name", ""))
        if lim is None:
            continue
        el = joint.find("limit")
        if el is None:
            raise SystemExit(f"{joint.get('name')} has no <limit> to set")
        el.set("lower", f"{lim[0]:g}")
        el.set("upper", f"{lim[1]:g}")
    missing = set(ALPHA_JOINT_LIMITS) - {j.get("name") for j in robot.findall("joint")}
    if missing:
        raise SystemExit(f"Alpha joints not found: {sorted(missing)}")
    log.append(f"set IsaacSwim joint limits on {len(ALPHA_JOINT_LIMITS)} arm joints")
    return log


def validate_tree(robot: ET.Element) -> str:
    links = {x.get("name") for x in robot.findall("link")}
    children, parents = {}, {}
    for j in robot.findall("joint"):
        p, c = j.find("parent"), j.find("child")
        if p is None or c is None:
            raise SystemExit(f"joint {j.get('name')} is missing a parent or child")
        pl, cl = p.get("link"), c.get("link")
        for ref in (pl, cl):
            if ref not in links:
                raise SystemExit(f"joint {j.get('name')} references unknown link {ref!r}")
        if cl in parents:
            raise SystemExit(f"link {cl!r} has two parents ({parents[cl]}, {j.get('name')})")
        parents[cl] = j.get("name")
        children.setdefault(pl, []).append(cl)

    roots = sorted(links - set(parents))
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link, found {len(roots)}: {roots}")

    seen, stack = set(), [roots[0]]
    while stack:
        node = stack.pop()
        if node in seen:
            raise SystemExit(f"cycle in the kinematic tree at {node!r}")
        seen.add(node)
        stack.extend(children.get(node, []))
    if seen != links:
        raise SystemExit(f"links unreachable from {roots[0]!r}: {sorted(links - seen)}")

    return roots[0]


def report(robot: ET.Element, thrusters: list[dict], root: str) -> str:
    links = robot.findall("link")
    joints = robot.findall("joint")
    movable = [j for j in joints if j.get("type") not in ("fixed", None)]

    total = 0.0
    for link in links:
        m = link.find("inertial/mass")
        if m is not None:
            total += float(m.get("value", 0.0))

    lines = [
        f"root link        {root}",
        f"links            {len(links)}",
        f"joints           {len(joints)}  ({len(movable)} movable, "
        f"{len(joints) - len(movable)} fixed)",
        f"total mass       {total:.4f} kg",
        f"displaced @1027  {total / 1027.0 * 1000:.3f} L for neutral buoyancy",
        "",
        "movable joints:",
    ]
    for j in movable:
        lim = j.find("limit")
        lo = float(lim.get("lower", 0.0)) if lim is not None else 0.0
        hi = float(lim.get("upper", 0.0)) if lim is not None else 0.0
        eff = float(lim.get("effort", 0.0)) if lim is not None else 0.0
        lines.append(
            f"  {j.get('name'):<30} {j.get('type'):<9} "
            f"[{lo:+.3f}, {hi:+.3f}] rad  effort {eff:g}"
        )

    lines += ["", f"thrusters ({len(thrusters)}), position and axis in base_link:"]
    for t in thrusters:
        px, py, pz = t["position"]
        ax, ay, az = t["axis"]
        lines.append(
            f"  {t['link']:<12} p ({px:+.4f} {py:+.4f} {pz:+.4f})"
            f"  a ({ax:+.3f} {ay:+.3f} {az:+.3f})  {'ccw' if t['ccw'] else 'cw '}"
        )

    if thrusters:
        A = allocation_matrix(thrusters)
        lines += ["", "allocation matrix A (6 x N), wrench = A @ thrust:"]
        for name, row in zip(("fx", "fy", "fz", "mx", "my", "mz"), A):
            lines.append("  " + name + "  " + "  ".join(f"{v:+.4f}" for v in row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the BlueROV2 Heavy and Alpha 5 URDF.")
    ap.add_argument("--jaws", choices=("direct", "mimic", "none"), default="direct",
                    help="how to handle the gripper's mimic chain")
    ap.add_argument("--mesh-format", choices=("obj", "keep"), default="obj",
                    help="transcode COLLADA and STL visuals to OBJ (default; "
                         "Isaac Sim 5.1 cannot import either) or keep them as-is")
    ap.add_argument("--thrusters", choices=("weld", "keep"), default="weld",
                    help="weld the propeller joints (default) or leave them continuous")
    ap.add_argument("--min-link-mass", type=float, default=0.01,
                    help="mass given to links the description leaves massless (kg)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--refetch", action="store_true", help="re-download the pinned files")
    ap.add_argument("--upstream-dir", type=Path,
                    help="use a local uvms-simulator checkout at the pinned commit instead "
                         "of downloading the vehicle; it is not changed")
    ap.add_argument("--reach-dir", type=Path,
                    help="use a local Robotic-Decision-Making-Lab/alpha checkout at the pinned "
                         "commit, including Reach's alpha_description/meshes/LICENSE")
    ap.add_argument("--report", action="store_true", help="print the summary and exit")
    args = ap.parse_args(argv)

    roots, provenance = {}, {}
    for source, local in ((UVMS, args.upstream_dir), (REACH, args.reach_dir)):
        roots[source.key], provenance[source.key] = source_input(
            source, local, force=args.refetch and local is None)
    if not args.report:
        validate_output_path([p for p, local in zip(roots.values(),
                                                    (args.upstream_dir, args.reach_dir)) if local],
                             args.out)

    with tempfile.TemporaryDirectory() as tmp:
        top = stage_description(roots, Path(tmp))
        staged = top.parent
        robot = ET.fromstring(run_xacro(top))

        footprint_log = drop_footprint(robot)
        alpha_log = adapt_reach_alpha(robot)
        jaw_log = rework_jaws(robot, args.jaws)
        dropped = strip_ros_only(robot)
        thruster_log = weld_thrusters(robot) if args.thrusters == "weld" else []
        massless = add_missing_inertials(robot, args.min_link_mass)
        thrusters = extract_thrusters(robot)
        root = validate_tree(robot)

        summary = report(robot, thrusters, root)
        if args.report:
            print(summary)
            return 0

        out = args.out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        n_meshes = rewrite_meshes(robot, staged, out, args.mesh_format)
        retain_notices(roots, out)
        for mesh_dir in sorted({p.parent for p in (out / "meshes" / "alpha").rglob("*")
                                if p.is_file()}):
            shutil.copy2(roots[REACH.key] / REACH_LICENSE, mesh_dir / "LICENSE")
        (out / "ASSET_SOURCES.json").write_text(json.dumps({
            "sources": {
                source.key: {
                    "provenance": provenance[source.key],
                    "repository": source.repo,
                    "commit": source.sha,
                    "files_sha256": source.files,
                } for source in SOURCES
            },
            "license": "Not covered by IsaacSwim's source license.  UVMS files: see "
                       "UPSTREAM_LICENSE.  Reach Alpha 5 description and meshes: see "
                       "REACH_LICENSE and REACH_NOTICE.txt.  All notices: SOURCE_NOTICES/.",
            "modifications": {
                "jaws": args.jaws, "mesh_format": args.mesh_format,
                "thrusters": args.thrusters, "min_link_mass": args.min_link_mass,
                "alpha_joint_limits": ALPHA_JOINT_LIMITS,
                "alpha_renames": ALPHA_RENAMES,
                "operations": ["compose stock BlueROV2 Heavy and Reach Alpha 5 (robot.xacro)",
                               "resolve xacro", "drop base_footprint", "rename arm frames",
                               "set arm joint limits", "remove ROS elements",
                               "weld propellers", "repair inertias",
                               "convert meshes when requested"],
            },
            "redistribution": "Do not redistribute the original or converted Reach files "
                              "(REACH_NOTICE.txt).  Retaining notices alone does not clear "
                              "redistribution of the other sources' files.",
        }, indent=2) + "\n")

    ET.indent(robot, space="  ")
    urdf = out / "bluerov2_heavy_alpha.urdf"
    generated_from = f"{UVMS.repo}@{UVMS.sha[:8]} and {REACH.repo}@{REACH.sha[:8]}"
    urdf.write_bytes(
        b'<?xml version="1.0"?>\n'
        + f"<!-- generated by isaacswim.assets.bluerov.build_urdf from {generated_from}; "
          "do not edit or redistribute: see REACH_NOTICE.txt -->\n".encode()
        + ET.tostring(robot, encoding="utf-8")
    )
    (out / "thrusters.json").write_text(json.dumps({
        "frame": "base_link",
        "source": f"{UVMS.repo}@{UVMS.sha} blue/xacro/bluerov2_heavy (stock, 45 deg)",
        "note": "axis is the propeller spin axis; thrust sign is a wiring convention, "
                "calibrate it before trusting the allocation",
        "thrusters": thrusters,
        "allocation_matrix": {
            "rows": ["fx", "fy", "fz", "mx", "my", "mz"],
            "columns": [t["link"] for t in thrusters],
            "A": allocation_matrix(thrusters),
        },
    }, indent=2) + "\n")

    for line in footprint_log + alpha_log:
        print(f"  {line}", file=sys.stderr)
    for line in jaw_log:
        print(f"  jaws: {line}", file=sys.stderr)
    for line in thruster_log:
        print(f"  thrusters: {line}", file=sys.stderr)
    if dropped:
        print(f"  dropped {len(dropped)} ROS-only elements: {sorted(set(dropped))}",
              file=sys.stderr)
    if massless:
        print(f"  gave inertia to {len(massless)} massless links: {', '.join(massless)}",
              file=sys.stderr)
    print(f"  copied {n_meshes} meshes", file=sys.stderr)
    print(f"\nwrote {urdf}\n", file=sys.stderr)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

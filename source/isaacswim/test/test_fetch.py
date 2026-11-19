from __future__ import annotations

import urllib.error

import pytest

FIXTURE = {"LICENSE": b"Reach license fixture", "meshes/arm.stl": b"solid arm"}


def fake_source(builder, files=FIXTURE, notices=("LICENSE",)):
    import dataclasses
    import hashlib

    return dataclasses.replace(
        builder.REACH, key="fixture", repo="example/fixture", sha="0" * 40,
        files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        notices=tuple(notices))


def serve(monkeypatch, builder, payloads):
    import io

    calls = []

    def urlopen(url):
        calls.append(url)
        rel = url.split("/", 6)[6]
        if rel not in payloads:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return io.BytesIO(payloads[rel])

    monkeypatch.setattr(builder.urllib.request, "urlopen", urlopen)
    return calls


def test_pinned_download_fetches_only_the_manifest_and_caches(tmp_path, monkeypatch):
    from isaacswim.assets.bluerov import build_urdf as builder

    source = fake_source(builder)
    calls = serve(monkeypatch, builder, {**FIXTURE, "other/arm_copy.stl": b"x"})
    root = builder.fetch_pinned(source, cache=tmp_path)
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {
        "LICENSE", "meshes/arm.stl", ".pinned"}
    assert len(calls) == 2 and not any("arm_copy" in c for c in calls)
    assert builder.fetch_pinned(source, cache=tmp_path) == root
    assert len(calls) == 2


def test_download_without_the_license_leaves_no_cache(tmp_path, monkeypatch):
    from isaacswim.assets.bluerov import build_urdf as builder

    source = fake_source(builder)
    serve(monkeypatch, builder, {"meshes/arm.stl": FIXTURE["meshes/arm.stl"]})
    with pytest.raises(urllib.error.HTTPError):
        builder.fetch_pinned(source, cache=tmp_path)
    assert not (tmp_path / source.key).exists()
    assert not (tmp_path / f"{source.key}.partial").exists()


def test_download_that_does_not_match_its_pin_is_rejected(tmp_path, monkeypatch):
    from isaacswim.assets.bluerov import build_urdf as builder

    source = fake_source(builder)
    serve(monkeypatch, builder, {**FIXTURE, "meshes/arm.stl": b"edited upstream"})
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        builder.fetch_pinned(source, cache=tmp_path)
    assert not (tmp_path / source.key).exists()


def test_cache_missing_its_license_is_fetched_again(tmp_path, monkeypatch):
    from isaacswim.assets.bluerov import build_urdf as builder

    source = fake_source(builder)
    calls = serve(monkeypatch, builder, FIXTURE)
    root = builder.fetch_pinned(source, cache=tmp_path)
    (root / "LICENSE").unlink()
    builder.fetch_pinned(source, cache=tmp_path)
    assert len(calls) == 4
    assert (root / "LICENSE").read_bytes() == FIXTURE["LICENSE"]


def test_local_checkout_must_hold_the_pinned_files_and_is_not_modified(tmp_path):
    from isaacswim.assets.bluerov import build_urdf as builder

    source = fake_source(builder)
    for rel, data in FIXTURE.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    root, provenance = builder.source_input(source, tmp_path)
    assert root == tmp_path.resolve() and provenance.startswith("local checkout")
    assert (tmp_path / "LICENSE").read_bytes() == FIXTURE["LICENSE"]

    with pytest.raises(ValueError, match="cannot be combined"):
        builder.source_input(source, tmp_path, force=True)
    (tmp_path / "LICENSE").unlink()
    with pytest.raises(ValueError, match="missing LICENSE"):
        builder.source_input(source, tmp_path)


def test_vehicle_source_never_downloads_an_alpha_and_reach_source_has_its_license():
    from isaacswim.assets.bluerov import build_urdf as builder

    assert not [f for f in builder.UVMS.files if "alpha" in f.lower()]
    assert builder.REACH.repo == "Robotic-Decision-Making-Lab/alpha"
    assert builder.REACH_LICENSE in builder.REACH.files
    assert builder.REACH_LICENSE in builder.REACH.notices
    meshes = [f for f in builder.REACH.files if f.endswith(".stl")]
    assert len(meshes) == 8
    assert all(f.startswith("alpha_description/meshes/") for f in meshes)


def test_output_keeps_reach_license_verbatim_with_the_disclaimer(tmp_path):
    from isaacswim.assets.bluerov import build_urdf as builder

    roots = {}
    for source in builder.SOURCES:
        root = roots[source.key] = tmp_path / source.key
        for rel in source.notices:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(f"{source.key} {rel} fixture")
    out = tmp_path / "out"
    out.mkdir()
    builder.retain_notices(roots, out)
    reach = (roots["reach-alpha"] / builder.REACH_LICENSE).read_bytes()
    assert (out / "REACH_LICENSE").read_bytes() == reach
    assert (out / "SOURCE_NOTICES/reach-alpha" / builder.REACH_LICENSE).read_bytes() == reach
    assert (out / "UPSTREAM_LICENSE").read_text() == "uvms-simulator LICENSE fixture"
    notice = (out / "REACH_NOTICE.txt").read_text()
    for phrase in ("not affiliated", "Simulation only", "not representative",
                   "Do not redistribute", "revoke", builder.REACH.sha):
        assert phrase in notice


@pytest.mark.parametrize("relative", [".", "generated", ".."])
def test_local_build_cannot_overwrite_its_input(tmp_path, relative):
    from isaacswim.assets.bluerov import build_urdf as builder

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        builder.validate_output_path([checkout], checkout / relative)


REACH_FRAGMENT = """<robot>
  <link name="base_link"/><link name="base_footprint"/>
  <joint name="base_footprint_joint" type="fixed">
    <parent link="base_link"/><child link="base_footprint"/></joint>
  <link name="alpha_m1_link"/><link name="alpha_ee_base_link"/>
  <link name="alpha_standard_jaws_base_link"/><link name="alpha_standard_jaws_tool"/>
  <joint name="alpha_axis_e" type="revolute"><parent link="base_link"/>
    <child link="alpha_m1_link"/><limit lower="0" upper="6.1"/></joint>
  <joint name="alpha_axis_d" type="revolute"><parent link="base_link"/>
    <child link="alpha_m1_link"/><limit lower="-3.49" upper="3.49"/></joint>
  <joint name="alpha_axis_c" type="revolute"><parent link="base_link"/>
    <child link="alpha_m1_link"/><limit lower="0" upper="3.22"/></joint>
  <joint name="alpha_axis_b" type="revolute"><parent link="base_link"/>
    <child link="alpha_m1_link"/><limit lower="0" upper="3.22"/></joint>
  <joint name="alpha_ee_joint" type="fixed"><parent link="alpha_m1_link"/>
    <child link="alpha_ee_base_link"/></joint>
  <joint name="alpha_standard_jaws_base_joint" type="fixed">
    <parent link="alpha_ee_base_link"/><child link="alpha_standard_jaws_base_link"/></joint>
  <joint name="alpha_standard_jaws_tool_joint" type="fixed">
    <parent link="alpha_standard_jaws_base_link"/><child link="alpha_standard_jaws_tool"/></joint>
</robot>"""


def test_reach_frames_get_isaacswim_names_limits_and_no_orphan_tool():
    import xml.etree.ElementTree as ET

    from isaacswim.assets.bluerov import build_urdf as builder

    robot = ET.fromstring(REACH_FRAGMENT)
    builder.drop_footprint(robot)
    builder.adapt_reach_alpha(robot)
    links = {x.get("name") for x in robot.findall("link")}
    joints = {x.get("name"): x for x in robot.findall("joint")}
    assert "base_footprint" not in links and "alpha_standard_jaws_tool" not in links
    assert {"alpha_jaws_base_link", "alpha_standard_jaws_base_link"} <= links
    assert {"alpha_jaws_joint", "alpha_standard_jaws_joint"} <= set(joints)
    assert joints["alpha_jaws_joint"].find("child").get("link") == "alpha_jaws_base_link"
    for name, (lo, hi) in builder.ALPHA_JOINT_LIMITS.items():
        lim = joints[name].find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == (lo, hi)


def test_lfs_pointer_is_followed_to_its_content(tmp_path, monkeypatch):
    import io

    from isaacswim.assets.bluerov import build_urdf as builder

    blob = b"#usda 1.0\n(real content)"
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 9\n"
    source = fake_source(builder, {"LICENSE": b"BSD", "a.usd": blob})

    def urlopen(url):
        if url.endswith("LICENSE"):
            return io.BytesIO(b"BSD")
        return io.BytesIO(blob if "media.githubusercontent.com" in url else pointer)

    monkeypatch.setattr(builder.urllib.request, "urlopen", urlopen)
    root = builder.fetch_pinned(source, cache=tmp_path)
    assert (root / "a.usd").read_bytes() == blob


def test_gripper_comes_from_robotiq_with_both_licenses():
    from isaacswim.assets.bluerov import attach_robotiq as robotiq

    src = robotiq.ROBOTIQ
    assert src.repo == "robotiq/isaacsim_assets" and len(src.sha) == 40
    assert set(src.notices) <= set(src.files)
    assert "grippers/Robotiq_2F_85/PACKAGE-LICENSES/LICENSE" in src.notices
    assert robotiq.ROBOTIQ_USD in src.files


def test_gripper_output_keeps_licenses_attribution_and_changes(tmp_path):
    from isaacswim.assets.bluerov import attach_robotiq as robotiq

    root = tmp_path / "src"
    for rel, text in (("LICENSE", "BSD 3-Clause fixture"),
                      ("grippers/Robotiq_2F_85/PACKAGE-LICENSES/LICENSE", "CC BY 4.0 fixture")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    out = tmp_path / "out"
    out.mkdir()
    robotiq.retain_robotiq_notices(root, out)
    assert (out / "ROBOTIQ_LICENSE").read_text() == "BSD 3-Clause fixture"
    assert (out / "ROBOTIQ_CC-BY-4.0.txt").read_text() == "CC BY 4.0 fixture"
    text = (out / "ROBOTIQ_ASSET_SOURCE.txt").read_text()
    for phrase in ("NVIDIA Corporation", "ROBOTIQ", "IsaacSwim's changes", "root_joint",
                   robotiq.ROBOTIQ.sha):
        assert phrase in text


GRIPPER_USDA = """#usda 1.0
(
    defaultPrim = "Robotiq_2F_85"
)
def Xform "Robotiq_2F_85" (
    variants = {
        string Fingertip = "Other"
        string Physics = "None"
    }
    prepend variantSets = ["Fingertip", "Physics"]
)
{
    def Xform "Robotiq_2F_85" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {
        bool physxArticulation:enabledSelfCollisions = 0
        def Xform "base_link" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {
        }
        def Scope "Joints"
        {
            def PhysicsFixedJoint "root_joint"
            {
                rel physics:body1 = </Robotiq_2F_85/Robotiq_2F_85/base_link>
            }
        }
    }
    variantSet "Fingertip" = {
        "Other" {
        }
        "Standard" {
        }
    }
    variantSet "Physics" = {
        "None" {
        }
        "Physx_parallel_grip" {
            over "Robotiq_2F_85"
            {
                over "Joints"
                {
                    def PhysicsRevoluteJoint "finger_joint"
                    {
                        rel physics:body0 = </Robotiq_2F_85/Robotiq_2F_85/base_link>
                    }
                }
            }
        }
    }
}
"""


def test_flatten_selects_variants_and_detaches_from_the_world(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    from isaacswim.assets.bluerov import attach_robotiq as robotiq

    src = tmp_path / "Robotiq_2F_85.usda"
    src.write_text(GRIPPER_USDA)
    before = src.read_bytes()
    out = tmp_path / "flat.usd"
    robotiq.flatten_robotiq(src, out)
    assert src.read_bytes() == before
    st = Usd.Stage.Open(str(out))
    root = st.GetPrimAtPath("/Robotiq_2F_85/Robotiq_2F_85")
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not [a for a in root.GetPropertyNames() if a.startswith("physxArticulation:")]
    assert st.GetPrimAtPath("/Robotiq_2F_85/Robotiq_2F_85/robotiq_base_link")
    assert not st.GetPrimAtPath("/Robotiq_2F_85/Robotiq_2F_85/Joints/root_joint")
    finger = UsdPhysics.Joint(st.GetPrimAtPath("/Robotiq_2F_85/Robotiq_2F_85/Joints/finger_joint"))
    assert [str(p).rsplit("/", 1)[-1] for p in finger.GetBody0Rel().GetTargets()] == [
        "robotiq_base_link"]


def test_red_stock_faces_are_baked_dark(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    from isaacswim.assets.bluerov import build_urdf as builder

    scene = trimesh.Scene()
    for name, colour, x in (("red", [255, 0, 0, 255], 0.0), ("grey", [128, 128, 128, 255], 2.0)):
        box = trimesh.creation.box()
        material = trimesh.visual.material.PBRMaterial(baseColorFactor=colour)
        box.visual = trimesh.visual.TextureVisuals(material=material)
        box.apply_translation([x, 0, 0])
        scene.add_geometry(box, node_name=name, geom_name=name)
    path = tmp_path / "hull.glb"
    scene.export(path)
    obj = builder.transcode_mesh(path, bake_colours=True)
    rows = [line.split() for line in obj.read_text().splitlines() if line.startswith("v ")]
    colours = np.array([[float(c) for c in row[4:7]] for row in rows])
    assert not (colours == [1.0, 0.0, 0.0]).all(axis=1).any()
    assert (np.abs(colours - 26 / 255) < 0.01).all(axis=1).any()
    assert (np.abs(colours - 128 / 255) < 0.01).all(axis=1).any()

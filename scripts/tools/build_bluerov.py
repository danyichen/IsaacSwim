"""Download the pinned sources and build the BlueROV2 Heavy vehicle."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

from isaacswim.assets import cache_dir
from isaacswim.launch import prepare_launch

parser = argparse.ArgumentParser(description="Build the BlueROV2 Heavy vehicle.")
parser.add_argument("--refetch", action="store_true", help="re-download the pinned files")
parser.add_argument("--upstream-dir", type=str,
                    help="use a local uvms-simulator checkout at the pinned commit")
parser.add_argument("--reach-dir", type=str,
                    help="use a local Robotic-Decision-Making-Lab/alpha checkout at the pinned "
                         "commit, including Reach's license")
parser.add_argument("--skip-urdf", action="store_true",
                    help="reuse the URDF already in the cache")
parser.add_argument("--no-robotiq", action="store_true", help="stop after the jaws vehicle")
parser.add_argument("--collider", choices=("convex_hull", "convex_decomposition"),
                    default="convex_decomposition")
parser.add_argument("--self-collision", action="store_true")
parser.add_argument("--stiffness", type=float, default=100.0,
                    help="position-drive stiffness authored on the arm joints")
parser.add_argument("--damping", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

alpha = cache_dir() / "bluerov2_heavy_alpha"
robotiq = cache_dir() / "bluerov2_heavy_robotiq"

if not args.skip_urdf:
    from isaacswim.assets.bluerov import build_urdf

    argv = ["--out", str(alpha)] + (["--refetch"] if args.refetch else [])
    if args.upstream_dir:
        argv += ["--upstream-dir", args.upstream_dir]
    if args.reach_dir:
        argv += ["--reach-dir", args.reach_dir]
    if build_urdf.main(argv):
        raise SystemExit("build_urdf failed")

prepare_launch(args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

urdf = alpha / f"{alpha.name}.urdf"
if not urdf.is_file():
    raise SystemExit(f"{urdf} not found -- run without --skip-urdf")
fields = UrdfConverterCfg.__dataclass_fields__
if "collision_type" in fields:
    collider = {"collision_type": {"convex_hull": "Convex Hull",
                                   "convex_decomposition": "Convex Decomposition"}[args.collider],
                "run_asset_transformer": False}
else:
    collider = {"collider_type": args.collider}
cfg = UrdfConverterCfg(
    asset_path=str(urdf),
    usd_dir=str(alpha),
    usd_file_name=f"{alpha.name}.usd",
    force_usd_conversion=True,
    fix_base=False,
    merge_fixed_joints=True,
    **collider,
    self_collision=args.self_collision,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=args.stiffness, damping=args.damping,
        ),
    ),
)
usd_path = UrdfConverter(cfg).usd_path
want = alpha / f"{alpha.name}.usd"
if os.path.realpath(usd_path) != os.path.realpath(want):
    from pxr import Usd
    Usd.Stage.Open(str(usd_path)).Flatten().Export(str(want))
    usd_path = want

from isaacswim.assets.bluerov.usd_cleanup import remove_dangling_references  # noqa: E402

removed = remove_dangling_references(want)
if removed:
    print(f"dropped {removed} dangling visual reference(s)")
print(f"\nUSD: {usd_path}", flush=True)

if not args.no_robotiq:
    from isaacswim.assets.bluerov import attach_robotiq

    attach_robotiq.main(["--src", str(alpha), "--out", str(robotiq)])

print("next: scripts/tools/check_bluerov.py, in air, before any water\n", flush=True)
simulation_app.close()

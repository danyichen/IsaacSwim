"""The BlueROV2 Heavy with a Reach Alpha 5 arm and Robotiq gripper."""

from __future__ import annotations

from isaacswim.assets import cache_dir

from .articulation import gripper_shapes, robot_cfg
from .cfg import ARM_HOME, SEABED_Z, build_hydro
from .controllers import BLUEROV2_DAMPING, AlphaIK, BatchStationKeeper, HoldPose, StationKeeper

JAWS_USD = cache_dir() / "bluerov2_heavy_alpha" / "bluerov2_heavy_alpha.usd"

DEFAULT_USD = cache_dir() / "bluerov2_heavy_robotiq" / "bluerov2_heavy_robotiq.usda"

__all__ = ["DEFAULT_USD", "JAWS_USD", "ARM_HOME", "SEABED_Z", "build_hydro", "robot_cfg",
           "gripper_shapes", "AlphaIK", "BLUEROV2_DAMPING", "BatchStationKeeper", "HoldPose",
           "StationKeeper"]

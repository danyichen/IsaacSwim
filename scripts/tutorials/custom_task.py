"""A complete custom task and scene: hover beside a marker buoy.

python -m isaacswim_bench.run --task scripts.tutorials.custom_task:BuoyHover \\
    --plant kinematic --episodes 5 --out runs/buoy
"""

from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import wrap_pi
from isaacswim_bench.core import (
    Command,
    CurrentSpec,
    Expert,
    Prop,
    SceneSpec,
    StepResult,
    Task,
    heading_frame,
    hold_timer,
    yaw_quat,
)
from isaacswim_bench.platform import ARM_STOW

STANDOFF = 1.0
POSITION_TOLERANCE = 0.15
HEADING_TOLERANCE = math.radians(10.0)
HOLD_SECONDS = 3.0


class BuoyHover(Task):
    name = "buoy_hover"
    horizon = 45.0
    seabed_z = -2.0
    summary = "hover one metre from a marker buoy, facing it"
    templates = ("Hover one metre from the orange buoy and face it.",)

    def sample(self, rng):
        bearing = rng.uniform(-0.6, 0.6)
        distance = rng.uniform(2.5, 3.5)
        speed = rng.uniform(0.0, 0.15)
        heading = rng.uniform(-math.pi, math.pi)
        return {
            "buoy": [distance * math.cos(bearing), distance * math.sin(bearing), -1.0],
            "spawn": [0.0, 0.0, -1.0],
            "spawn_yaw": float(rng.uniform(-0.3, 0.3)),
            "current": [speed * math.cos(heading), speed * math.sin(heading), 0.0],
        }

    def scene(self, params):
        return SceneSpec(
            seabed_z=self.seabed_z,
            props=[Prop("buoy", "sphere", (0.2,), color=(1.0, 0.42, 0.05), reflectivity=2.0)],
            poses={"buoy": (tuple(params["buoy"]), yaw_quat(0.0))},
            spawn_pos=tuple(params["spawn"]),
            spawn_yaw=params["spawn_yaw"],
            current=CurrentSpec(mean=tuple(params["current"])),
            arm=dict(ARM_STOW),
        )

    def dress(self, stage, *, seed=0, root="/World", origin=(0.0, 0.0, 0.0)):
        from pxr import Gf, UsdGeom

        path = f"{root}/survey_marker"
        marker = UsdGeom.Cone.Define(stage, path)
        marker.CreateRadiusAttr(0.3)
        marker.CreateHeightAttr(0.6)
        marker.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.8, 0.1)])
        position = Gf.Vec3d(origin[0], origin[1], origin[2] + self.seabed_z + 0.3)
        marker.AddTranslateOp().Set(position)
        return (path,)

    def label_sonar(self, osim, stage, roots):
        for path in roots:
            osim.set_sonar_reflectivity(path, 2.0)

    def target(self, params):
        buoy = np.asarray(params["buoy"], dtype=np.float64)
        offset = buoy - np.asarray(params["spawn"], dtype=np.float64)
        direction = offset / np.linalg.norm(offset)
        return buoy - STANDOFF * direction, math.atan2(direction[1], direction[0])

    def reset(self, params, world):
        return {}

    def evaluate(self, world, params, mem, dt):
        position, yaw = self.target(params)
        error = float(np.linalg.norm(world.pos - position))
        heading_error = self.yaw_error(world, yaw)
        held = error < POSITION_TOLERANCE and heading_error < HEADING_TOLERANCE
        timer = hold_timer(mem, "hold", held, dt)
        return StepResult(reward=-error - 0.3 * heading_error, success=timer >= HOLD_SECONDS)

    def metrics(self, world, params, mem):
        position, _ = self.target(params)
        return {"final_error_m": float(np.linalg.norm(world.pos - position))}

    def privileged(self, world, params):
        position, yaw = self.target(params)
        return np.array([*heading_frame(position - world.pos, world.yaw),
                         wrap_pi(yaw - world.yaw)], dtype=np.float32)

    def expert(self, params, arm):
        return HoverExpert(*self.target(params))


class HoverExpert(Expert):
    def __init__(self, position, yaw):
        super().__init__()
        self.position, self.yaw = position, yaw
        self.state = "hover"

    def act(self, world):
        return Command(pos=self.position.copy(), yaw=self.yaw, arm=dict(ARM_STOW))

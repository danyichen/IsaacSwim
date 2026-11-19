from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import wrap_pi

from ..core import (
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
from ..platform import ARM_STOW

BAY_Z_ABOVE_SEABED = 1.0
INNER_W = 0.84
DEPTH = 1.0
ROOF = 0.24
WALL_T = 0.04
FLARE_LEN = 0.6
FLARE_ANGLE = math.radians(35.0)
TOL = np.array([0.12, 0.08, 0.10])
YAW_TOL = math.radians(8.0)
HOLD_S = 3.0
SETTLED_SPEED = 0.08


class Dock(Task):
    name = "dock"
    horizon = 110.0
    seabed_z = -1.5
    summary = "enter a 0.84 m docking garage across a cross-current and settle"
    video_view = {"eye": (-1.7, -0.5, 0.12), "look": (0.8, 0.0, -0.02),  # noqa: RUF012
                  "follow_yaw": True}
    templates = (
        "Dock the vehicle inside the garage.",
        "Fly into the docking station and stop inside it.",
        "Return to the dock: enter the garage between the guide plates and hold.",
    )

    @property
    def bay(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.seabed_z + BAY_Z_ABOVE_SEABED])

    def sample(self, rng):
        cross = rng.uniform(0.0, 0.4) * rng.choice([-1.0, 1.0])
        return {"spawn": [float(rng.uniform(-4.0, -3.0)), float(rng.uniform(-1.0, 1.0)),
                          float(self.bay[2] + rng.uniform(-0.3, 0.3))],
                "spawn_yaw": float(rng.uniform(-0.7, 0.7)),
                "current": [float(rng.uniform(-0.15, 0.15)), float(cross), 0.0],
                "gust": float(rng.uniform(0.0, 0.15))}

    def scene(self, params):
        z0, zb = self.seabed_z, self.bay[2]
        top = zb + ROOF
        h = top - z0
        grey, yellow = (0.30, 0.31, 0.33), (0.95, 0.75, 0.10)
        yw = INNER_W / 2 + WALL_T / 2
        props = [
            Prop("roof", "box", (DEPTH + 0.1, INNER_W + 2 * WALL_T, 0.04), color=grey,
                 reflectivity=2.0),
            Prop("wall_left", "box", (DEPTH, WALL_T, h), color=grey, reflectivity=2.0),
            Prop("wall_right", "box", (DEPTH, WALL_T, h), color=grey, reflectivity=2.0),
            Prop("wall_back", "box", (WALL_T, INNER_W + 2 * WALL_T, h), color=grey,
                 reflectivity=2.0),
            Prop("flare_left", "box", (FLARE_LEN, WALL_T, h), color=yellow,
                 reflectivity=2.0),
            Prop("flare_right", "box", (FLARE_LEN, WALL_T, h), color=yellow,
                 reflectivity=2.0),
            Prop("light_left", "box", (0.05, 0.05, 0.05), color=(0.1, 1.0, 0.2),
                 emissive=(0.1, 0.9, 0.2), kind="visual"),
            Prop("light_right", "box", (0.05, 0.05, 0.05), color=(0.1, 1.0, 0.2),
                 emissive=(0.1, 0.9, 0.2), kind="visual"),
            Prop("target_mark", "box", (0.01, 0.25, 0.25), color=(0.95, 0.95, 0.95),
                 emissive=(0.4, 0.4, 0.4), kind="visual"),
        ]
        xm = -DEPTH / 2
        c, s = math.cos(FLARE_ANGLE), math.sin(FLARE_ANGLE)
        poses = {
            "roof": ((0.0, 0.0, top + 0.02), yaw_quat(0.0)),
            "wall_left": ((0.0, yw, z0 + h / 2), yaw_quat(0.0)),
            "wall_right": ((0.0, -yw, z0 + h / 2), yaw_quat(0.0)),
            "wall_back": ((DEPTH / 2 + WALL_T / 2, 0.0, z0 + h / 2), yaw_quat(0.0)),
            "flare_left": ((xm - FLARE_LEN / 2 * c, yw + FLARE_LEN / 2 * s, z0 + h / 2),
                           yaw_quat(-FLARE_ANGLE)),
            "flare_right": ((xm - FLARE_LEN / 2 * c, -yw - FLARE_LEN / 2 * s, z0 + h / 2),
                            yaw_quat(FLARE_ANGLE)),
            "light_left": ((xm, yw, top - 0.05), yaw_quat(0.0)),
            "light_right": ((xm, -yw, top - 0.05), yaw_quat(0.0)),
            "target_mark": ((DEPTH / 2 - 0.01, 0.0, zb), yaw_quat(0.0)),
        }
        return SceneSpec(seabed_z=z0, props=props, poses=poses,
                         spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
                         current=CurrentSpec(mean=tuple(params["current"]),
                                             sigma=(params["gust"],) * 2 + (0.0,)),
                         arm=dict(ARM_STOW), camera_tilt=0.0)

    def reset(self, params, world):
        return {"entry_speed": None, "inside": False}

    def evaluate(self, world, params, mem, dt):
        err = world.pos - self.bay
        yerr = self.yaw_error(world, 0.0)
        inside = world.pos[0] > -DEPTH / 2 and abs(world.pos[1]) < INNER_W / 2
        if inside and not mem["inside"] and mem["entry_speed"] is None:
            mem["entry_speed"] = world.speed
        mem["inside"] = inside
        ok = bool(np.all(np.abs(err) < TOL)) and yerr < YAW_TOL and world.speed < SETTLED_SPEED
        held = hold_timer(mem, "hold", ok, dt)
        return StepResult(reward=-float(np.linalg.norm(err)) - 0.3 * yerr
                          + (1.0 if ok else 0.0),
                          success=held >= HOLD_S,
                          info={"err": err.tolist(), "yaw_err": yerr, "inside": inside})

    def metrics(self, world, params, mem):
        return {"entry_speed": mem["entry_speed"],
                "final_err_m": float(np.linalg.norm(world.pos - self.bay)),
                "cross_current": float(params["current"][1])}

    def privileged(self, world, params):
        return np.array([*heading_frame(self.bay - world.pos, world.yaw),
                         wrap_pi(0.0 - world.yaw)], dtype=np.float32)

    def expert(self, params, arm):
        return _DockExpert(self.bay)


class _DockExpert(Expert):
    STANDOFF_X = -1.7
    RUN_SPEED = 0.10
    ALIGN_Y = 0.08
    ABORT_Y = 0.18
    DEADLINES = {"align": 40.0, "run_in": 30.0}

    def __init__(self, bay):
        super().__init__()
        self.bay = bay
        self.x = self.STANDOFF_X
        self.state = "align"

    def act(self, world):
        zb = self.bay[2]
        stow = dict(ARM_STOW)
        if self.state == "align":
            goal = np.array([self.STANDOFF_X, 0.0, zb])
            e = world.pos - goal
            if (abs(e[0]) < 0.12 and abs(e[1]) < self.ALIGN_Y and abs(e[2]) < 0.06
                    and abs(wrap_pi(world.yaw)) < math.radians(5.0)
                    and world.speed < SETTLED_SPEED) or self.expired():
                self.x = self.STANDOFF_X
                self.goto("run_in", world.t)
            return Command(pos=goal, yaw=0.0, arm=stow)
        if self.state == "run_in":
            outside = world.pos[0] < -DEPTH / 2
            if outside and abs(world.pos[1]) > self.ABORT_Y:
                self.goto("align", world.t)
                return Command(pos=np.array([self.STANDOFF_X, 0.0, zb]), yaw=0.0, arm=stow)
            self.x = min(self.x + self.RUN_SPEED * self.dt, self.bay[0])
            if self.x >= self.bay[0]:
                self.goto("settle", world.t)
            return Command(pos=np.array([self.x, 0.0, zb]), yaw=0.0, arm=stow)
        return Command(pos=self.bay.copy(), yaw=0.0, arm=stow)

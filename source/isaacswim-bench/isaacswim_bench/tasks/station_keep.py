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
    World,
    heading_frame,
    hold_timer,
    yaw_quat,
)
from ..platform import ARM_STOW

STANDOFF = 1.0
HOLD_S = 10.0
POS_TOL = 0.10
YAW_TOL = math.radians(8.0)
PANEL_Z = 1.0


class StationKeep(Task):
    name = "station_keep"
    horizon = 60.0
    seabed_z = -1.5
    summary = "hold 1 m off an inspection panel for 10 s in a gusting current"
    templates = (
        "Hold position one meter in front of the yellow inspection panel, facing it.",
        "Hover facing the yellow panel, about a meter away, and hold steady.",
        "Station-keep one meter off the face of the inspection panel.",
    )

    def sample(self, rng):
        dist = rng.uniform(2.6, 3.4)
        bearing = rng.uniform(-0.5, 0.5)
        panel_xy = np.array([dist * math.cos(bearing), dist * math.sin(bearing)])
        panel_yaw = wrap_pi(bearing + math.pi + rng.uniform(-0.35, 0.35))
        speed = rng.uniform(0.15, 0.4)
        heading = rng.uniform(-math.pi, math.pi)
        return {
            "panel_xy": panel_xy.tolist(), "panel_yaw": float(panel_yaw),
            "spawn": [float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)),
                      float(self.seabed_z + rng.uniform(0.9, 1.3))],
            "spawn_yaw": float(rng.uniform(-0.6, 0.6)),
            "current": [speed * math.cos(heading), speed * math.sin(heading), 0.0],
            "gust": float(rng.uniform(0.06, 0.15)),
        }

    def target(self, params) -> tuple[np.ndarray, float]:
        yaw = params["panel_yaw"]
        n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        face = np.array([*params["panel_xy"], self.seabed_z + PANEL_Z]) + 0.05 * n
        return face + STANDOFF * n, wrap_pi(yaw + math.pi)

    def scene(self, params):
        z0 = self.seabed_z
        xy, yaw = np.asarray(params["panel_xy"]), params["panel_yaw"]
        n = np.array([math.cos(yaw), math.sin(yaw)])
        t = np.array([-n[1], n[0]])
        q = yaw_quat(yaw)
        props = [
            Prop("panel", "box", (0.10, 0.80, 0.70), color=(0.85, 0.70, 0.12),
                 reflectivity=2.0),
            Prop("panel_frame", "box", (0.12, 0.50, PANEL_Z - 0.35), color=(0.25, 0.25, 0.27)),
            Prop("gauge", "cylinder", (0.07, 0.03), color=(0.95, 0.95, 0.95),
                 axis="X", kind="visual", emissive=(0.3, 0.3, 0.3)),
            Prop("marker", "box", (0.01, 0.20, 0.20), color=(0.05, 0.05, 0.05),
                 kind="visual"),
        ]
        poses = {
            "panel": ((*xy, z0 + PANEL_Z), q),
            "panel_frame": ((*xy, z0 + (PANEL_Z - 0.35) / 2), q),
            "gauge": ((*(xy + 0.06 * n + 0.22 * t), z0 + PANEL_Z + 0.2), q),
            "marker": ((*(xy + 0.055 * n), z0 + PANEL_Z), q),
        }
        return SceneSpec(
            seabed_z=z0, props=props, poses=poses,
            spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
            current=CurrentSpec(mean=tuple(params["current"]),
                                sigma=(params["gust"],) * 2 + (params["gust"] / 3,)),
            arm=dict(ARM_STOW), camera_tilt=-0.15)

    def reset(self, params, world):
        return {"errs": []}

    def evaluate(self, world, params, mem, dt):
        p, yaw = self.target(params)
        err = float(np.linalg.norm(world.pos - p))
        yerr = self.yaw_error(world, yaw)
        ok = err < POS_TOL and yerr < YAW_TOL
        held = hold_timer(mem, "hold", ok, dt)
        if held > 0.0:
            mem["errs"].append(err)
        return StepResult(reward=-err - 0.3 * yerr + (1.0 if ok else 0.0),
                          success=held >= HOLD_S,
                          info={"pos_err": err, "yaw_err": yerr, "held": held})

    def metrics(self, world, params, mem):
        e = np.asarray(mem["errs"] or [np.nan])
        return {"hold_err_rms_m": float(np.sqrt(np.mean(e ** 2))),
                "hold_err_max_m": float(np.max(e)),
                "current_speed": float(np.linalg.norm(params["current"]))}

    def privileged(self, world, params):
        p, yaw = self.target(params)
        return np.array([*heading_frame(p - world.pos, world.yaw),
                         wrap_pi(yaw - world.yaw)], dtype=np.float32)

    def expert(self, params, arm):
        return _GoAndHold(*self.target(params))


class _GoAndHold(Expert):
    def __init__(self, pos, yaw):
        super().__init__()
        self.pos, self.yaw = pos, yaw

    def act(self, world: World) -> Command:
        d = self.pos - world.pos
        far = np.linalg.norm(d[:2]) > 1.0
        yaw = math.atan2(d[1], d[0]) if far else self.yaw
        self.state = "transit" if far else "hold"
        return Command(pos=self.pos, yaw=yaw, arm=dict(ARM_STOW))

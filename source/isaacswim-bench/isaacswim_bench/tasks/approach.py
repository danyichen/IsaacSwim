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

OBJECTS = {
    "buoy": ("orange", (1.00, 0.42, 0.05), "sphere", (0.18,)),
    "barrel": ("yellow", (0.95, 0.80, 0.10), "cylinder", (0.16, 0.36)),
    "crate": ("red", (0.80, 0.10, 0.08), "box", (0.34, 0.34, 0.30)),
    "toolbox": ("blue", (0.10, 0.30, 0.85), "box", (0.44, 0.24, 0.20)),
}
NAMES = tuple(OBJECTS)
STANDOFF = (0.5, 1.2)
BEARING_TOL = math.radians(15.0)
ALT_BAND = (0.6, 1.6)
HOLD_S = 2.0


def _height(name) -> float:
    _, _, shape, size = OBJECTS[name]
    if shape == "sphere":
        return 2 * size[0]
    return size[1] if shape == "cylinder" else size[2]


class TargetApproach(Task):
    name = "target_approach"
    horizon = 60.0
    seabed_z = -1.5
    summary = "find the named object among four and hover in front of it"
    templates = (
        "Go to the {color} {object} and hover in front of it.",
        "Find the {color} {object} and stop about a meter away, facing it.",
        "Fly to the {object} and hold position in front of it.",
        "Approach the {color} one and hover facing it.",
    )

    def sample(self, rng):
        for _ in range(200):
            bearings = np.sort(rng.uniform(-1.1, 1.1, size=len(NAMES)))
            ranges = rng.uniform(2.0, 3.8, size=len(NAMES))
            xy = np.stack([ranges * np.cos(bearings), ranges * np.sin(bearings)], 1)
            d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(NAMES)) * 9
            if d.min() > 1.3:
                break
        order = rng.permutation(len(NAMES))
        layout = {NAMES[i]: xy[k].tolist() for k, i in enumerate(order)}
        target = NAMES[int(rng.integers(len(NAMES)))]
        speed = rng.uniform(0.0, 0.4)
        h = rng.uniform(-math.pi, math.pi)
        return {"layout": layout, "yaws": {n: float(rng.uniform(-math.pi, math.pi))
                                           for n in NAMES},
                "target": target, "object": target, "color": OBJECTS[target][0],
                "spawn": [0.0, 0.0, float(self.seabed_z + rng.uniform(0.9, 1.2))],
                "spawn_yaw": float(rng.uniform(-0.3, 0.3)),
                "current": [speed * math.cos(h), speed * math.sin(h), 0.0]}

    def scene(self, params):
        z0 = self.seabed_z
        props, poses = [], {}
        for n in NAMES:
            _, rgb, shape, size = OBJECTS[n]
            props.append(Prop(n, shape, size, color=rgb, reflectivity=2.0))
            poses[n] = ((*params["layout"][n], z0 + _height(n) / 2),
                        yaw_quat(params["yaws"][n]))
        return SceneSpec(seabed_z=z0, props=props, poses=poses,
                         spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
                         current=CurrentSpec(mean=tuple(params["current"]),
                                             sigma=(0.09, 0.09, 0.0)),
                         arm=dict(ARM_STOW), camera_tilt=-0.3)

    @staticmethod
    def in_front(world: World, xy) -> bool:
        d = np.asarray(xy) - world.pos[:2]
        r = float(np.linalg.norm(d))
        bearing = abs(wrap_pi(math.atan2(d[1], d[0]) - world.yaw))
        return (STANDOFF[0] <= r <= STANDOFF[1] and bearing < BEARING_TOL
                and ALT_BAND[0] <= world.altitude <= ALT_BAND[1] and world.speed < 0.1)

    def evaluate(self, world, params, mem, dt):
        tgt = params["target"]
        success = False
        failure = None
        for n in NAMES:
            held = hold_timer(mem, n, self.in_front(world, params["layout"][n]), dt)
            if held >= HOLD_S:
                if n == tgt:
                    success = True
                else:
                    failure = f"wrong_object:{n}"
        r = float(np.linalg.norm(np.asarray(params["layout"][tgt]) - world.pos[:2]))
        return StepResult(reward=-0.2 * max(r - 0.85, 0.0) + (1.0 if mem[tgt] > 0 else 0.0),
                          success=success, failure=failure, info={"range": r})

    def metrics(self, world, params, mem):
        r = float(np.linalg.norm(np.asarray(params["layout"][params["target"]])
                                 - world.pos[:2]))
        return {"final_range_m": r, "target": params["target"]}

    def privileged(self, world, params):
        rows = []
        for n in NAMES:
            rel = heading_frame(np.array([*params["layout"][n], self.seabed_z])
                                - world.pos, world.yaw)
            rows += [*rel, 1.0 if n == params["target"] else 0.0]
        return np.asarray(rows, dtype=np.float32)

    def expert(self, params, arm):
        return _Approach(self, params)


class _Approach(Expert):
    STANDOFF = 0.85

    def __init__(self, task, params):
        super().__init__()
        self.xy = np.asarray(params["layout"][params["target"]], dtype=np.float64)
        self.z = task.seabed_z + 1.0
        self.goal = None
        self.hold = None
        self.state = "turn"

    def act(self, world):
        d = self.xy - world.pos[:2]
        face = math.atan2(d[1], d[0])
        if self.goal is None:
            self.goal = self.xy - self.STANDOFF * d / np.linalg.norm(d)
            self.hold = np.array([*world.pos[:2], self.z])
        if self.state == "turn":
            if abs(wrap_pi(face - world.yaw)) < 0.15:
                self.goto("approach", world.t)
            return Command(pos=self.hold, yaw=face,
                           arm=dict(ARM_STOW))
        return Command(pos=np.array([*self.goal, self.z]), yaw=face, arm=dict(ARM_STOW))

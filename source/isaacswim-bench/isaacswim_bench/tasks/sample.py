from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import quat_apply_inverse

from ..core import (
    CurrentSpec,
    Prop,
    SceneSpec,
    StepResult,
    Task,
    heading_frame,
    hold_timer,
    yaw_quat,
)
from ..platform import ARM_STOW, CAMERA_TILT_WORK, GRASP_DEPTH, grip_angle_for_gap
from .manip import HOVER, ToolExpert

CUBE = 0.045
CUBE_DENSITY = 1.05 * 1025.0
COLORS = {"red": (0.85, 0.12, 0.08), "yellow": (0.95, 0.80, 0.10),
          "green": (0.15, 0.70, 0.25)}
BASKET_IN = 0.22
BASKET_WALL = (0.012, 0.05)
FLOOR_T = 0.01
DROP_H = 0.08


class SampleCollect(Task):
    name = "sample_collect"
    horizon = 150.0
    sample_height = CUBE
    sample_width = CUBE
    grasp_depth = GRASP_DEPTH
    video_view = {"eye": (1.20, -1.55, 0.52), "look": (0.26, 0.0, -0.26),  # noqa: RUF012
                  "follow_yaw": False}
    seabed_z = -0.55
    summary = "pick the named 45 mm sample and place it in the basket"
    keeper_ki = 100.0
    templates = (
        "Pick up the {color} sample and put it in the basket.",
        "Collect the {color} sample into the basket.",
        "Place the {color} cube in the sample basket.",
        "Grab the {color} rock sample and drop it in the collection basket.",
    )

    def sample(self, rng):
        side = rng.choice([-1.0, 1.0])
        basket = [float(rng.uniform(1.0, 1.2)), float(side * rng.uniform(0.30, 0.42))]
        pts = []
        while len(pts) < len(COLORS):
            p = np.array([rng.uniform(0.92, 1.28), rng.uniform(-0.40, 0.40)])
            if np.linalg.norm(p - basket) < 0.30:
                continue
            if all(np.linalg.norm(p - q) > 0.14 for q in pts):
                pts.append(p)
        speed = rng.uniform(0.0, 0.30)
        h = rng.uniform(-math.pi, math.pi)
        target = str(rng.choice(list(COLORS)))
        return {"samples": {c: pts[k].tolist() for k, c in enumerate(COLORS)},
                "sample_yaws": {c: float(rng.uniform(-0.6, 0.6)) for c in COLORS},
                "basket": basket, "basket_yaw": float(rng.uniform(-0.4, 0.4)),
                "target": target, "color": target,
                "spawn": [float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1)),
                          float(rng.uniform(-0.05, 0.05))],
                "spawn_yaw": float(rng.uniform(-0.25, 0.25)),
                "current": [speed * math.cos(h), speed * math.sin(h), 0.0]}

    def scene(self, params):
        z0 = self.seabed_z
        props, poses = [], {}
        for c, rgb in COLORS.items():
            n = f"sample_{c}"
            props.append(Prop(n, "box", (CUBE,) * 3, color=rgb, kind="dynamic",
                              density=CUBE_DENSITY, friction=(1.4, 1.2), graspable=True))
            poses[n] = ((*params["samples"][c], z0 + CUBE / 2 + 0.001),
                        yaw_quat(params["sample_yaws"][c]))
        bx, by = params["basket"]
        byaw = params["basket_yaw"]
        cb, sb = math.cos(byaw), math.sin(byaw)
        t, h = BASKET_WALL
        wall_c = (BASKET_IN + t) / 2
        props.append(Prop("basket_floor", "box", (BASKET_IN + 2 * t,) * 2 + (FLOOR_T,),
                          color=(0.20, 0.22, 0.25)))
        poses["basket_floor"] = ((bx, by, z0 + FLOOR_T / 2), yaw_quat(byaw))
        for k, (dx, dy, long_x) in enumerate(((wall_c, 0, False), (-wall_c, 0, False),
                                              (0, wall_c, True), (0, -wall_c, True))):
            size = (BASKET_IN + 2 * t, t, h) if long_x else (t, BASKET_IN, h)
            props.append(Prop(f"basket_wall_{k}", "box", size, color=(0.95, 0.55, 0.10)))
            poses[f"basket_wall_{k}"] = ((bx + cb * dx - sb * dy, by + sb * dx + cb * dy,
                                          z0 + FLOOR_T + h / 2), yaw_quat(byaw))
        return SceneSpec(seabed_z=z0, props=props, poses=poses,
                         spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
                         current=CurrentSpec(mean=tuple(params["current"]),
                                             sigma=(0.06, 0.06, 0.0)),
                         arm=dict(ARM_STOW), camera_tilt=CAMERA_TILT_WORK)

    def in_basket(self, params, p) -> bool:
        bx, by = params["basket"]
        rel = quat_apply_inverse(yaw_quat(params["basket_yaw"]),
                                 np.array([p[0] - bx, p[1] - by, 0.0]))
        lim = BASKET_IN / 2 - self.sample_width / 2 + 0.01
        return (abs(rel[0]) < lim and abs(rel[1]) < lim
                and p[2] < self.seabed_z + FLOOR_T + self.sample_height + 0.01)

    def evaluate(self, world, params, mem, dt):
        tgt = f"sample_{params['target']}"
        p = world.obj(tgt)
        for c in COLORS:
            if c != params["target"] and self.in_basket(params, world.obj(f"sample_{c}")):
                return StepResult(reward=-1.0, failure=f"wrong_sample:{c}")
        released = (not world.grip_closed
                    and float(np.linalg.norm(world.tool_pos - p)) > 0.05)
        ok = self.in_basket(params, p) and released
        held = hold_timer(mem, "hold", ok, dt)
        basket = np.array([*params["basket"], p[2]])
        reward = -float(np.linalg.norm(world.tool_pos - p)) \
            - float(np.linalg.norm(p - basket)) + (1.0 if ok else 0.0)
        return StepResult(reward=reward, success=held >= 1.0,
                          info={"sample_to_basket": float(np.linalg.norm((p - basket)[:2]))})

    def metrics(self, world, params, mem):
        p = world.obj(f"sample_{params['target']}")
        return {"place_err_m": float(np.linalg.norm(p[:2] - np.asarray(params["basket"]))),
                "target": params["target"]}

    def privileged(self, world, params):
        p = world.obj(f"sample_{params['target']}")
        b = np.array([*params["basket"], self.seabed_z])
        return np.array([*heading_frame(p - world.pos, world.yaw),
                         *heading_frame(b - world.pos, world.yaw),
                         *heading_frame(p - world.tool_pos, world.yaw)], dtype=np.float32)

    def expert(self, params, arm):
        return _PickPlace(self, params, arm)


class _PickPlace(ToolExpert):
    DESCEND_RATE = 0.08
    BOTTOM_SETTLE = 0.6
    GRIP_HOLD = 1.2
    MISSED_Q = grip_angle_for_gap(0.023)
    RETRIES = 3
    DEADLINES = {"settle": 4.0, "goto": 30.0, "descend": 16.0, "regrip": 6.0,
                 "lift": 5.0, "transit": 30.0, "release": 3.0}

    def __init__(self, task: SampleCollect, params, arm):
        super().__init__(arm)
        self.task, self.params = task, params
        self.name = f"sample_{params['target']}"
        self.latch = None
        self.state = "settle"
        self.misses = 0
        z0 = task.seabed_z
        self.carry_z = z0 + FLOOR_T + DROP_H + task.sample_height - task.grasp_depth
        self.drop_xy = np.asarray(params["basket"], dtype=np.float64)
        self.held_t = self.free_t = 0.0
        self.last_q = 0.0

    HELD_Q = 0.10
    DEBOUNCE = 0.3

    def holding(self, world, cube) -> bool:
        if world.grasped is not None:
            return world.grasped == self.name
        stopped = abs(world.grip_q - self.last_q) < 0.01
        self.last_q = world.grip_q
        return (world.grip_closed and stopped and self.HELD_Q < world.grip_q < self.MISSED_Q
                and float(np.linalg.norm(world.tool_pos - cube)) < 0.04)

    def sync(self, world):
        if self.state == "settle":
            return
        cube = world.obj(self.name)
        held = self.holding(world, cube)
        self.held_t = self.held_t + self.dt if held else 0.0
        self.free_t = 0.0 if held else self.free_t + self.dt
        st = self.state
        if self.held_t >= self.DEBOUNCE and st in ("goto", "descend", "regrip", "done"):
            self.grabbed = world.tool_pos.copy()
            self.goto("lift", world.t)
        elif self.free_t >= self.DEBOUNCE and st in ("lift", "transit"):
            over = self.task.in_basket(self.params, [*cube[:2], self.task.seabed_z])
            self.goto("done" if over else "regrip", world.t)
        elif (self.free_t >= self.DEBOUNCE and st == "done"
              and not self.task.in_basket(self.params, cube)
              and cube[2] < self.task.seabed_z + self.task.sample_height):
            self.goto("regrip", world.t)

    def act(self, world):
        st = self.state
        cube = world.obj(self.name)
        if st == "settle":
            self.hold_yaw = 0.0
            if self.arm_cmd is None:
                self.arm_cmd = dict(world.arm_q)
            if (self.state_t > 0.5 and abs(world.yaw) < 0.05) or self.expired():
                y = self.params["sample_yaws"][self.params["target"]]
                self.arm_cmd = self.plan_grasp_arm(world, (-math.sin(y), math.cos(y), 0.0))
                self.goto("goto", world.t)
            return self.cmd_hold(world, False)
        if st == "goto":
            self.latch = cube.copy()
            want = self.latch + np.array([0.0, 0.0, HOVER])
            ready = self.arm_arrived(world, 0.05) and self.jaws_open(world)
            if (ready and self.over(world, want, 0.015) and world.speed < 0.04
                    and self.state_t > 0.5) or self.expired():
                self.start_descent(world, "descend")
            return self.cmd_tool(world, want, False)
        if st == "descend":
            floor = self.latch[2] + self.task.sample_height / 2 - self.task.grasp_depth
            cmd, settled = self.descend(world, cube, floor)
            if settled or self.expired():
                self.goto("close", world.t)
            return cmd
        if st == "close":
            if self.state_t > self.GRIP_HOLD:
                if world.grip_q > self.MISSED_Q and self.misses < self.RETRIES:
                    self.misses += 1
                    self.goto("regrip", world.t)
                else:
                    self.grabbed = world.tool_pos.copy()
                    self.goto("lift", world.t)
            return self.cmd_hold(world, True)
        if st == "regrip":
            want = np.array([*world.tool_pos[:2], cube[2] + HOVER])
            clear = self.jaws_open(world) and world.tool_pos[2] > want[2] - 0.03
            if clear or self.expired():
                self.goto("goto", world.t)
            return self.cmd_tool(world, want, False)
        if st == "lift":
            z = min(self.grabbed[2] + self.DESCEND_RATE * self.state_t, self.carry_z)
            want = np.array([self.grabbed[0], self.grabbed[1], z])
            risen = z >= self.carry_z and world.tool_pos[2] > self.carry_z - 0.015
            if risen or self.expired():
                self.goto("transit", world.t)
            return self.cmd_tool(world, want, True)
        if st == "transit":
            want = np.array([*self.drop_xy, self.carry_z])
            if (self.over(world, want, 0.02) and world.speed < 0.05
                    and self.state_t > 0.5) or self.expired():
                self.goto("release", world.t)
            return self.cmd_tool(world, want, True)
        if st == "release":
            if (self.jaws_open(world) and self.state_t > 0.5) or self.expired():
                self.goto("done", world.t)
            return self.cmd_hold(world, False)
        return self.cmd_hold(world, False)

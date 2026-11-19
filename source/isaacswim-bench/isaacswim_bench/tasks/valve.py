from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import wrap_pi

from ..core import (
    CurrentSpec,
    Joint,
    Prop,
    SceneSpec,
    StepResult,
    Task,
    heading_frame,
    hold_timer,
    yaw_quat,
)
from ..platform import ARM_STOW, CAMERA_TILT_WORK, GRASP_DEPTH, jaw_direction_yaw
from .manip import CLOSE_HOLD, ToolExpert

MANIFOLD = (0.50, 0.50, 0.25)
HANDLE = (0.16, 0.03, 0.03)
HANDLE_GAP = 0.02
GOAL = math.radians(90.0)
SUCCESS_AT = math.radians(80.0)
WRONG_WAY = math.radians(30.0)


class ValveTurn(Task):
    name = "valve_turn"
    horizon = 120.0
    video_view = {"eye": (1.20, -1.55, 0.52), "look": (0.26, 0.0, -0.26),  # noqa: RUF012
                  "follow_yaw": False}
    seabed_z = -0.55
    summary = "grasp a T-bar valve and turn it 90 degrees the instructed way"
    templates = (
        "Turn the valve handle a quarter turn {direction}.",
        "Rotate the valve ninety degrees {direction}.",
        "Grip the T-handle on the manifold and turn it {direction} by a quarter turn.",
    )

    @property
    def valve_xy(self) -> np.ndarray:
        return np.array([1.10, 0.0])

    @property
    def handle_z(self) -> float:
        return self.seabed_z + MANIFOLD[2] + HANDLE_GAP + HANDLE[2] / 2

    def sample(self, rng):
        sign = float(rng.choice([-1.0, 1.0]))
        speed = rng.uniform(0.0, 0.30)
        h = rng.uniform(-math.pi, math.pi)
        return {"theta0": float(rng.uniform(-0.7, 0.7)), "sign": sign,
                "direction": "counterclockwise" if sign > 0 else "clockwise",
                "friction": float(rng.uniform(0.3, 1.2)),
                "spawn": [float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.15, 0.15)),
                          float(rng.uniform(0.15, 0.3))],
                "spawn_yaw": float(rng.uniform(-0.25, 0.25)),
                "current": [speed * math.cos(h), speed * math.sin(h), 0.0]}

    def scene(self, params):
        z0 = self.seabed_z
        x, y = self.valve_xy
        props = [
            Prop("manifold", "box", MANIFOLD, color=(0.22, 0.24, 0.28), reflectivity=2.0),
            Prop("stem", "cylinder", (0.012, HANDLE_GAP + 0.01), color=(0.6, 0.6, 0.6),
                 kind="visual"),
            Prop("valve_plate", "cylinder", (0.10, 0.004), color=(0.85, 0.85, 0.80),
                 kind="visual"),
            Prop("handle", "box", HANDLE, color=(0.90, 0.15, 0.10), kind="dynamic",
                 density=2000.0, friction=(1.2, 1.0), graspable=True),
        ]
        poses = {
            "manifold": ((x, y, z0 + MANIFOLD[2] / 2), yaw_quat(0.0)),
            "stem": ((x, y, z0 + MANIFOLD[2] + HANDLE_GAP / 2), yaw_quat(0.0)),
            "valve_plate": ((x, y, z0 + MANIFOLD[2] + 0.002), yaw_quat(0.0)),
            "handle": ((x, y, self.handle_z), yaw_quat(params["theta0"])),
        }
        return SceneSpec(
            seabed_z=z0, props=props, poses=poses,
            joints=[Joint("valve", "handle", (x, y, self.handle_z), "Z",
                          friction_torque=params["friction"], damping=0.05)],
            spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
            current=CurrentSpec(mean=tuple(params["current"]), sigma=(0.06, 0.06, 0.0)),
            arm=dict(ARM_STOW), camera_tilt=CAMERA_TILT_WORK)

    def reset(self, params, world):
        return {"last": world.joints["valve"], "turned": 0.0, "peak": 0.0}

    def evaluate(self, world, params, mem, dt):
        a = world.joints["valve"]
        mem["turned"] += wrap_pi(a - mem["last"])
        mem["last"] = a
        progress = params["sign"] * mem["turned"]
        mem["peak"] = max(mem["peak"], progress)
        if progress < -WRONG_WAY:
            return StepResult(reward=-1.0, failure="wrong_direction",
                              info={"progress": progress})
        handle = world.obj("handle")
        released = (not world.grip_closed
                    and float(np.linalg.norm(world.tool_pos - handle)) > 0.05)
        ok = progress >= SUCCESS_AT and released
        held = hold_timer(mem, "hold", ok, dt)
        reward = -float(np.linalg.norm(world.tool_pos - handle)) + progress / GOAL \
            + (1.0 if ok else 0.0)
        return StepResult(reward=reward, success=held >= 1.0,
                          info={"progress_deg": math.degrees(progress)})

    def metrics(self, world, params, mem):
        return {"turned_deg": math.degrees(params["sign"] * mem["turned"]),
                "friction_Nm": params["friction"], "direction": params["direction"]}

    def privileged(self, world, params):
        h = world.obj("handle")
        a = world.joints["valve"]
        return np.array([*heading_frame(h - world.pos, world.yaw),
                         math.sin(a - world.yaw), math.cos(a - world.yaw),
                         params["sign"],
                         *heading_frame(h - world.tool_pos, world.yaw)], dtype=np.float32)

    def expert(self, params, arm):
        return _TurnValve(self, params, arm)


class _TurnValve(ToolExpert):
    DEADLINES = {"settle": 6.0, "deploy": 14.0, "goto": 34.0, "descend": 26.0,
                 "turn": 12.0, "release": 3.0, "withdraw": 8.0}
    OVERSHOOT = math.radians(6.0)
    GOTO_TOL = 0.010
    GOTO_SPEED = 0.04

    def __init__(self, task: ValveTurn, params, arm):
        super().__init__(arm)
        self.task, self.params = task, params
        self.b_goal = None
        self.state = "settle"
        self._last_valve = None
        self._turned = 0.0
        self._retry = False
        self._regrasps = 0

    def _plan(self, world):
        th = world.joints["valve"]
        across = (-math.sin(th), math.cos(th), 0.0)
        q = self.plan_grasp_arm(world, across)
        lo, hi = self.arm.limits["alpha_axis_b"]
        for b in (q["alpha_axis_b"], q["alpha_axis_b"] + math.pi,
                  q["alpha_axis_b"] - math.pi):
            if not lo <= b <= hi:
                continue
            qb = {**q, "alpha_axis_b": b}
            y0 = jaw_direction_yaw(self.arm, qb, world.quat)
            y1 = jaw_direction_yaw(self.arm, {**qb, "alpha_axis_b": b + 0.05}, world.quat)
            sgn = 1.0 if wrap_pi(y1 - y0) > 0 else -1.0
            remaining = max(0.0, GOAL - self.params["sign"] * self._turned)
            goal = b + self.params["sign"] * sgn * (remaining + self.OVERSHOOT)
            if lo <= goal <= hi:
                self.b_goal = goal
                return qb
        raise RuntimeError("no wrist roll leaves room for a quarter turn")

    def act(self, world):
        angle = world.joints["valve"]
        if self._last_valve is not None:
            self._turned += wrap_pi(angle - self._last_valve)
        self._last_valve = angle
        progress = self.params["sign"] * self._turned
        st = self.state
        handle = world.obj("handle")
        if st == "settle":
            self.hold_yaw = 0.0
            if self.arm_cmd is None:
                self.arm_cmd = dict(world.arm_q)
            if (self.state_t > 1.5 and abs(world.yaw) < 0.05) or self.expired():
                self.arm_cmd = self._plan(world)
                self.goto("deploy", world.t)
            return self.cmd_hold(world, False)
        if st == "deploy":
            if (self.arm_arrived(world) and self.jaws_open(world)) or self.expired():
                self.goto("goto", world.t)
            return self.cmd_hold(world, False)
        if st == "goto":
            want = handle + np.array([0.0, 0.0, 0.10])
            if (self.over(world, want, self.GOTO_TOL) and world.speed < self.GOTO_SPEED
                    and self.state_t > 2.0) or self.expired():
                self.latch = handle.copy()
                self.start_descent(world, "descend")
            return self.cmd_tool(world, want, False)
        if st == "descend":
            floor = self.latch[2] + HANDLE[2] / 2 - GRASP_DEPTH
            cmd, settled = self.descend(world, self.latch, floor)
            if settled or self.expired():
                self.goto("close", world.t)
            return cmd
        if st == "close":
            if self.state_t > CLOSE_HOLD:
                self.goto("turn", world.t)
            return self.cmd_hold(world, True)
        if st == "turn":
            self.arm_cmd = {**self.arm_cmd, "alpha_axis_b": self.b_goal}
            done = abs(world.arm_q["alpha_axis_b"] - self.b_goal) < 0.03
            if progress >= GOAL or (done and self.state_t > 1.0) or self.expired():
                self._retry = progress < SUCCESS_AT
                self.goto("release", world.t)
            return self.cmd_hold(world, True)
        if st == "release":
            if (self.jaws_open(world) and self.state_t > 1.5) or self.expired():
                self.descend_from = float(world.tool_pos[2])
                self.goto("withdraw", world.t)
            return self.cmd_hold(world, False)
        if st == "withdraw":
            want = np.array([self.latch[0], self.latch[1],
                             min(self.descend_from + 0.04 * self.state_t,
                                 self.descend_from + 0.12)])
            if self.state_t > 0.12 / 0.04 + 1.0:
                if self._retry and self._regrasps < 2:
                    self._regrasps += 1
                    self.arm_cmd = self._plan(world)
                    self.goto("deploy", world.t)
                else:
                    self.goto("done", world.t)
            return self.cmd_tool(world, want, False)
        return self.cmd_hold(world, False)

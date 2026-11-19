from __future__ import annotations

import math

import numpy as np

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

N_SEG = 4
SEG_LEN = 1.5
PIPE_R = 0.10
ALTITUDE = 1.0
ALT_BAND = (0.6, 1.5)
LATERAL_TOL = 0.4
COVERAGE = 0.85
END_TOL = 0.5
TOO_LOW = 0.5


class PipelineInspect(Task):
    name = "pipeline_inspect"
    horizon = 90.0
    seabed_z = -1.5
    summary = "fly a 6 m pipeline with random bends to its end marker"
    templates = (
        "Follow the pipeline to the orange marker at its end, staying about one meter above it.",
        "Inspect the pipeline: fly along it until you reach the orange end marker.",
        "Track the pipe on the seabed all the way to the orange marker.",
    )

    def sample(self, rng):
        start = np.array([rng.uniform(0.8, 1.2), rng.uniform(-0.5, 0.5)])
        h = rng.uniform(-0.35, 0.35)
        heads = [h]
        for _ in range(N_SEG - 1):
            h = h + rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.6)
            heads.append(float(np.clip(h, -1.0, 1.0)))
        speed = rng.uniform(0.0, 0.4)
        cur_h = rng.uniform(-math.pi, math.pi)
        return {"start": start.tolist(), "headings": [float(v) for v in heads],
                "spawn": [0.0, float(rng.uniform(-0.3, 0.3)),
                          float(self.seabed_z + rng.uniform(0.9, 1.2))],
                "spawn_yaw": float(rng.uniform(-0.4, 0.4)),
                "current": [speed * math.cos(cur_h), speed * math.sin(cur_h), 0.0],
                "gust": float(rng.uniform(0.0, 0.18))}

    def vertices(self, params) -> np.ndarray:
        pts = [np.asarray(params["start"], dtype=np.float64)]
        for h in params["headings"]:
            pts.append(pts[-1] + SEG_LEN * np.array([math.cos(h), math.sin(h)]))
        return np.stack(pts)

    def samples(self, params, step: float = 0.1) -> np.ndarray:
        v = self.vertices(params)
        out = []
        for a, b in zip(v[:-1], v[1:], strict=True):
            n = int(round(np.linalg.norm(b - a) / step))
            out += [a + (b - a) * k / n for k in range(n)]
        out.append(v[-1])
        return np.stack(out)

    def point_at(self, params, s: float) -> tuple[np.ndarray, float]:
        v = self.vertices(params)
        s = min(max(s, 0.0), SEG_LEN * N_SEG)
        k = min(int(s // SEG_LEN), N_SEG - 1)
        h = params["headings"][k]
        return v[k] + (s - k * SEG_LEN) * np.array([math.cos(h), math.sin(h)]), h

    def project(self, params, xy) -> tuple[float, float]:
        v = self.vertices(params)
        best = (0.0, math.inf)
        for k, (a, b) in enumerate(zip(v[:-1], v[1:], strict=True)):
            ab = b - a
            u = float(np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            d = float(np.linalg.norm(xy - (a + u * ab)))
            if d < best[1]:
                best = (k * SEG_LEN + u * SEG_LEN, d)
        return best

    def scene(self, params):
        z0 = self.seabed_z
        v = self.vertices(params)
        props, poses = [], {}
        for k, h in enumerate(params["headings"]):
            mid = (v[k] + v[k + 1]) / 2
            props.append(Prop(f"pipe_{k}", "cylinder", (PIPE_R, SEG_LEN + 0.02),
                              axis="X", color=(0.35, 0.33, 0.30), reflectivity=2.5))
            poses[f"pipe_{k}"] = ((*mid, z0 + PIPE_R), yaw_quat(h))
        for k in range(1, N_SEG):
            props.append(Prop(f"flange_{k}", "cylinder", (PIPE_R * 1.35, 0.08), axis="X",
                              color=(0.55, 0.50, 0.40), kind="visual"))
            poses[f"flange_{k}"] = ((*v[k], z0 + PIPE_R), yaw_quat(params["headings"][k]))
        props.append(Prop("end_marker", "box", (0.25, 0.25, 0.45), color=(1.0, 0.45, 0.05),
                          emissive=(0.35, 0.12, 0.0), reflectivity=3.0))
        poses["end_marker"] = ((*(v[-1] + 0.3 * (v[-1] - v[-2]) / SEG_LEN), z0 + 0.225),
                               yaw_quat(params["headings"][-1]))
        return SceneSpec(seabed_z=z0, props=props, poses=poses,
                         spawn_pos=tuple(params["spawn"]), spawn_yaw=params["spawn_yaw"],
                         current=CurrentSpec(mean=tuple(params["current"]),
                                             sigma=(params["gust"],) * 2 + (0.0,)),
                         arm=dict(ARM_STOW), camera_tilt=-0.45)

    def reset(self, params, world):
        return {"seen": np.zeros(len(self.samples(params)), dtype=bool), "path": 0.0,
                "last": world.pos.copy()}

    def _covered(self, params, world) -> np.ndarray:
        pts = self.samples(params)
        near = np.linalg.norm(pts - world.pos[:2], axis=1) < LATERAL_TOL
        alt_ok = ALT_BAND[0] <= world.altitude <= ALT_BAND[1]
        return near & alt_ok

    def evaluate(self, world, params, mem, dt):
        before = mem["seen"].sum()
        mem["seen"] |= self._covered(params, world)
        mem["path"] += float(np.linalg.norm(world.pos - mem["last"]))
        mem["last"] = world.pos.copy()
        cov = float(mem["seen"].mean())
        end = self.vertices(params)[-1]
        d_end = float(np.linalg.norm(world.pos[:2] - end))
        _, lateral = self.project(params, world.pos[:2])
        at_end = d_end < END_TOL and ALT_BAND[0] <= world.altitude <= ALT_BAND[1] \
            and world.speed < 0.15
        held = hold_timer(mem, "hold", at_end, dt)
        failure = "too_low" if world.altitude < TOO_LOW else None
        success = held >= 1.0 and cov >= COVERAGE
        reward = 0.2 * (mem["seen"].sum() - before) - 0.1 * lateral \
            - 0.2 * abs(world.altitude - ALTITUDE)
        return StepResult(reward=float(reward), success=success, failure=failure,
                          info={"coverage": cov, "lateral": lateral, "to_end": d_end})

    def metrics(self, world, params, mem):
        return {"coverage": float(mem["seen"].mean()), "path_m": mem["path"],
                "pipe_m": SEG_LEN * N_SEG}

    def privileged(self, world, params):
        s, lateral = self.project(params, world.pos[:2])
        p, h = self.point_at(params, s)
        rel = heading_frame(np.array([*p, self.seabed_z + ALTITUDE]) - world.pos, world.yaw)
        end = self.vertices(params)[-1]
        return np.array([*rel, math.sin(h - world.yaw), math.cos(h - world.yaw),
                         SEG_LEN * N_SEG - s, float(np.linalg.norm(end - world.pos[:2]))],
                        dtype=np.float32)

    def expert(self, params, arm):
        return _FollowPipe(self, params)


class _FollowPipe(Expert):
    LEAD = 0.35
    SPEED = 0.2

    def __init__(self, task: PipelineInspect, params):
        super().__init__()
        self.task, self.params = task, params
        self.s = 0.0
        self.z = task.seabed_z + ALTITUDE
        self.state = "acquire"

    def act(self, world: World) -> Command:
        L = SEG_LEN * N_SEG
        if self.state == "acquire":
            p, h = self.task.point_at(self.params, 0.0)
            d = np.linalg.norm(world.pos[:2] - p)
            if d < 0.2 and abs(world.pos[2] - self.z) < 0.15:
                self.goto("follow", world.t)
            far = d > 0.6
            yaw = math.atan2(p[1] - world.pos[1], p[0] - world.pos[0]) if far else h
            return Command(pos=np.array([*p, self.z]), yaw=yaw, arm=dict(ARM_STOW))
        s_now, _ = self.task.project(self.params, world.pos[:2])
        self.s = max(self.s, min(s_now + self.LEAD, self.s + self.SPEED * self.dt))
        self.s = min(self.s, L)
        p, h = self.task.point_at(self.params, self.s)
        if self.s >= L:
            self.state = "arrive"
        return Command(pos=np.array([*p, self.z]), yaw=h, arm=dict(ARM_STOW))

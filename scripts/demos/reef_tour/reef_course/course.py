from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from isaacswim.np_math import wrap_pi
from isaacswim_bench.core import (
    Command,
    CurrentSpec,
    Expert,
    SceneSpec,
    StepResult,
    Task,
    World,
    heading_frame,
)
from isaacswim_bench.platform import ARM_STOW
from isaacswim_bench.tasks.dock import Dock

from .service import SERVICE_PIPE, ConnectedPipeline, CoralRelocate, MeterRead, TourValve

Z0 = -1.5
DOCK_BAY = (11.6, -3.5, Z0 + 1.0)
UNDOCK = 2.2
UNDOCK_SPEED = 0.15
PIPE_START = (-10.0, -3.0)
PIPE_HEADINGS = (0.0, 0.3, 0.1, -0.3)
PANEL = ((4.6, -1.0), math.pi)
ARM_HANG = 0.53
TRANSIT_Z = Z0 + 1.65
TRANSIT_CLEAR = 0.2
TRANSIT_HALF_WIDTH = 0.6
PIPE_CURRENT = 0.30
PIPE_GUST = 0.06
SITE_MARGIN = 30.0
TRANSIT_SPEED = 0.15


def tall_sets() -> list[tuple[np.ndarray, float, float]]:
    out = [(np.array(PANEL[0]), 0.55, 1.37), (np.array([1., -1.]), 0.55, 0.32)]

    return out


class Site:
    def __init__(self, key, task, origin, start, *, pin=None, props=False):
        task.seabed_z = Z0
        self.key, self.task = key, task
        self.origin = np.array([*origin, 0.0])
        self.start = tuple(float(v) for v in start)
        self.pin = dict(pin or {})
        self.props = props

    @property
    def label(self) -> str:
        return self.task.name

    def sample(self, rng) -> dict:
        return {**self.task.sample(rng), **self.pin}

    def name(self, local: str) -> str:
        return f"{self.key}_{local}"

    def local(self, world: World) -> World:
        o = self.origin
        n = len(self.key) + 1
        own = {k[n:]: (p - o, q) for k, (p, q) in world.objects.items()
               if k.startswith(self.key + "_")}
        joints = {k[n:]: a for k, a in world.joints.items() if k.startswith(self.key + "_")}
        grasped = world.grasped[n:] if world.grasped and world.grasped.startswith(
            self.key + "_") else None
        return replace(world, pos=world.pos - o, tool_pos=world.tool_pos - o, objects=own,
                       joints=joints, grasped=grasped)

    def to_course(self, cmd: Command) -> Command:
        return replace(cmd, pos=np.asarray(cmd.pos, dtype=np.float64) + self.origin)

    def scene(self, params) -> SceneSpec:
        spec = self.task.scene(params)
        o = self.origin
        props = [replace(p, name=self.name(p.name)) for p in spec.props]
        poses = {self.name(k): (tuple(np.asarray(p) + o), q) for k, (p, q) in spec.poses.items()}
        joints = [replace(j, name=self.name(j.name), body=self.name(j.body),
                          anchor=tuple(np.asarray(j.anchor) + o)) for j in spec.joints]
        return replace(spec, props=props, poses=poses, joints=joints)


def _sites() -> list[Site]:
    h0 = PIPE_HEADINGS[0]
    lead = np.array(PIPE_START) - 1.2 * np.array([math.cos(h0), math.sin(h0)])
    bx, by = DOCK_BAY[:2]
    return [
        Site("pipe", ConnectedPipeline(), (0.0, 0.0), (*lead, Z0 + 1.0, h0),
             pin={"start": list(PIPE_START), "headings": list(PIPE_HEADINGS),
                  "vertices": SERVICE_PIPE}),
        Site("sample", CoralRelocate(), (-2.25, -1.1),
             (-0.4, -1.1, Z0 + ARM_HANG + 0.25, math.pi), props=True),
        Site("valve", TourValve(), (-0.1, -1.0), (0.69, -1.0, Z0 + 1.15, 0.0), props=True),
        Site("panel", MeterRead(), (0.0, 0.0), (1.6, -1.0, Z0 + 1.1, 0.0),
             pin={"panel_xy": list(PANEL[0]), "panel_yaw": PANEL[1]}, props=True),
        Site("dock", Dock(), (bx, by), (bx - 3.5, by, DOCK_BAY[2], 0.0)),
    ]


SITES = _sites()
SITES = [SITES[i] for i in [0, 2, 1, 3, 4]]


class CourseTour(Task):
    name = "course_tour"
    horizon = 900.0
    seabed_z = Z0
    summary = ("inspect and service the reef station, relocate a coral fragment, "
               "read the instrument station, and dock")
    video_view = {"eye": (-1.8, -0.9, 0.6), "look": (0.8, 0.0, -0.3),  # noqa: RUF012
                  "follow_yaw": True}
    templates = (
        "Leave the dock, inspect the circulation pipeline to its valve, "
        "turn the valve {direction}, "
        "move the coral fragment tagged {sample_color} into the nursery holder, hold steady "
        "at the instrument station and report its pressure, temperature, pH and current speed, "
        "then return to the dock.",
    )
    sites = SITES
    start_at_first = False
    report_fields = MeterRead.report_fields
    reef = None

    def dress(self, stage, *, seed=0, root="/World", origin=(0.0, 0.0, 0.0)):
        from .level import locate_reef
        from .scenery import TEXTURE_DIR
        from .scenery.course import dress

        level = locate_reef(self.reef) if self.reef and root == "/World" else None
        return dress(stage, TEXTURE_DIR, seed=seed, root=root, origin=origin, reef=level)

    def label_sonar(self, osim, stage, roots):
        from .scenery.course import label_sonar

        label_sonar(osim, stage, roots)

    @property
    def bay(self) -> np.ndarray:
        return np.array(DOCK_BAY, dtype=np.float64)

    def sample(self, rng):
        sub = {s.key: s.sample(rng) for s in self.sites}
        speed = rng.uniform(0.0, PIPE_CURRENT)
        h = rng.uniform(-math.pi, math.pi)
        sub["pipe"].update(current=[speed * math.cos(h), speed * math.sin(h), 0.0],
                           gust=PIPE_GUST)
        currents = {}
        for s in self.sites:
            c = s.task.scene(sub[s.key]).current
            currents[s.key] = {"mean": [float(v) for v in c.mean],
                               "sigma": [float(v) for v in c.sigma], "tau": float(c.tau)}
        return {"currents": currents, "sites": sub,
                "sample_color": sub["sample"]["color"],
                "direction": sub["valve"]["direction"]}

    def scene(self, params):
        props, poses, joints = [], {}, []
        for s in self.sites:
            if s.props:
                spec = s.scene(params["sites"][s.key])
                props += spec.props
                poses.update(spec.poses)
                joints += spec.joints
        spawn = self.sites[0].start if self.start_at_first else (*DOCK_BAY, 0.0)
        return SceneSpec(seabed_z=Z0, props=props, poses=poses, joints=joints,
                         spawn_pos=tuple(spawn[:3]), spawn_yaw=float(spawn[3]),
                         current=self._current(
                             params, self.sites[0 if self.start_at_first else -1].key),
                         arm=dict(ARM_STOW), camera_tilt=-0.35)

    @staticmethod
    def _current(params, key) -> CurrentSpec:
        c = params["currents"][key]
        return CurrentSpec(mean=tuple(c["mean"]), sigma=tuple(c["sigma"]), tau=c["tau"])

    def current(self, world, params, mem):
        s = self.sites[mem["next"]]
        to = np.asarray(s.start[:2])
        d0 = float(np.linalg.norm(mem["leg_from"] - to))
        d = float(np.linalg.norm(world.pos[:2] - to))
        f = 1.0 if d0 < 0.5 else float(np.clip(1.0 - d / d0, 0.0, 1.0))
        mem["ramp"] = f = max(mem["ramp"], f)
        f = f * f * (3.0 - 2.0 * f)
        a, b = self._current(params, mem["cur_from"]), self._current(params, s.key)
        mix = lambda u, v: tuple((1.0 - f) * p + f * q for p, q in zip(u, v, strict=True))  # noqa: E731
        return CurrentSpec(mean=mix(a.mean, b.mean), sigma=mix(a.sigma, b.sigma), tau=b.tau)

    def keeper_gain(self, world, params, mem):
        return self.sites[mem["next"]].task.keeper_ki

    def _leave(self, k, world, mem):
        mem["cur_from"] = self.sites[k - 1].key if k > 0 else self.sites[-1].key
        mem["leg_from"] = np.array(world.pos[:2], dtype=np.float64)
        mem["ramp"] = 0.0

    _progress = 0

    def _start(self, k, params, world, mem):
        s = self.sites[k]
        mem["site"] = s.task.reset(params["sites"][s.key], s.local(world))
        distance = float(np.linalg.norm(np.asarray(s.start[:2]) - mem["leg_from"]))
        mem["deadline"] = world.t + s.task.horizon + SITE_MARGIN + distance / TRANSIT_SPEED

    def reset(self, params, world):
        self._progress = 0
        mem = {"next": 0, "reached": [], "site_metrics": {}}
        self._leave(0, world, mem)
        self._start(0, params, world, mem)
        return mem

    def evaluate(self, world, params, mem, dt):
        k = mem["next"]
        s = self.sites[k]
        res = s.task.evaluate(s.local(world), params["sites"][s.key], mem["site"], dt)
        info = {"site": s.label, **res.info}
        if res.failure is not None:
            return StepResult(reward=-1.0, failure=f"{s.label}:{res.failure}", info=info)
        if not res.success and world.t > mem["deadline"]:
            return StepResult(reward=-1.0, failure=f"{s.label}:timeout", info=info)
        last = k == len(self.sites) - 1
        if res.success:
            mem["reached"].append((s.label, round(world.t, 2)))
            mem["site_metrics"][s.label] = s.task.metrics(
                s.local(world), params["sites"][s.key], mem["site"])
            if last:
                return StepResult(reward=res.reward + 1.0, success=True, info=info)
            mem["next"] = self._progress = k + 1
            self._leave(k + 1, world, mem)
            self._start(k + 1, params, world, mem)
        return StepResult(reward=res.reward + k, info=info)

    def metrics(self, world, params, mem):
        return {"sites_done": len(mem["reached"]), "done_at": mem["reached"],
                "site_metrics": mem["site_metrics"],
                "final_dock_err_m": float(np.linalg.norm(world.pos - self.bay))}

    def privileged(self, world, params):
        s = self.sites[min(self._progress, len(self.sites) - 1)]
        p = np.array(s.start[:3])
        return np.array([*heading_frame(p - world.pos, world.yaw),
                         wrap_pi(s.start[3] - world.yaw), self._progress / len(self.sites)],
                        dtype=np.float32)

    def expert(self, params, arm):
        return _TourExpert(self, params, arm)


class Leg:
    def __init__(self, pos, yaw, *, tol=0.15, speed=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.yaw, self.tol, self.speed = float(yaw), tol, speed


CLIMB_TOL = 0.25


def transit_z(a, b) -> float:
    a, b = np.asarray(a[:2], dtype=np.float64), np.asarray(b[:2], dtype=np.float64)
    ab = b - a
    z = TRANSIT_Z
    for xy, r, top in tall_sets():
        u = float(np.clip(np.dot(xy - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
        if np.linalg.norm(a + u * ab - xy) < r + TRANSIT_HALF_WIDTH:
            z = max(z, Z0 + top + TRANSIT_CLEAR + ARM_HANG + CLIMB_TOL)
    return z


SHORT_HOP = 1.5


def transit(here, yaw, start) -> list[Leg]:
    here = np.asarray(here, dtype=np.float64)
    goal = np.asarray(start[:3], dtype=np.float64)
    d = goal[:2] - here[:2]
    h = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 0.3 else yaw
    if np.linalg.norm(d) < SHORT_HOP:
        z = max(here[2], goal[2])
        legs = [Leg((*here[:2], z), yaw, tol=CLIMB_TOL)]
        if np.linalg.norm(d) > 0.8 and abs(wrap_pi(h - yaw)) > 0.8:
            legs.append(Leg((*here[:2], z), h, tol=0.3))
            yaw = h
        return legs + [Leg((*goal[:2], z), yaw, tol=0.2), Leg(goal, yaw, tol=0.15),
                       Leg(goal, start[3], tol=0.15)]
    z = transit_z(here, start)
    return [Leg((*here[:2], z), yaw, tol=CLIMB_TOL), Leg((*here[:2], z), h, tol=0.3),
            Leg((*goal[:2], z), h, tol=0.25), Leg(goal, h, tol=0.15),
            Leg(goal, start[3], tol=0.15)]


class _TourExpert(Expert):
    LEG_DEADLINE = 120.0

    def __init__(self, task: CourseTour, params, arm):
        super().__init__()
        self.task, self.params, self.arm = task, params, arm
        self.k = -1
        bay = np.array(DOCK_BAY)
        self.legs = [Leg(bay, 0.0, tol=0.1), Leg(bay - [UNDOCK, 0.0, 0.0], 0.0, tol=0.15,
                                                 speed=UNDOCK_SPEED)]
        self.leg = 0
        self.leg_t = 0.0
        self.sub: Expert | None = None
        self.mem: dict = {}
        self.state = "dock"
        self.detail = "undock"

    def _site_begin(self, world):
        self.k += 1
        s = self.task.sites[self.k]
        self.legs, self.leg, self.leg_t = transit(world.pos, world.yaw, s.start), 0, 0.0
        self.sub = None
        self.goto(s.label, world.t)
        self.detail = "transit"

    def act(self, world: World) -> Command:
        if self.k < 0 and self.task.start_at_first:
            self.k = 0
            self.goto(self.task.sites[0].label, world.t)
            self._hand_over(world)
        if self.k >= 0 and self.sub is not None:
            return self._run_site(world)
        leg = self.legs[self.leg]
        goal = leg.pos
        if leg.speed is not None:
            start = self.legs[self.leg - 1].pos
            d = leg.pos - start
            n = float(np.linalg.norm(d))
            goal = start + d * min(1.0, leg.speed * self.leg_t / max(n, 1e-9))
        self.leg_t += self.dt
        arrived = (float(np.linalg.norm(world.pos - leg.pos)) < leg.tol
                   and abs(wrap_pi(leg.yaw - world.yaw)) < math.radians(8.0))
        if arrived or self.leg_t > self.LEG_DEADLINE:
            if not arrived:
                self.timeouts.append(f"{self.state} leg {self.leg}")
            self.leg, self.leg_t = self.leg + 1, 0.0
            if self.leg == len(self.legs):
                if self.k < 0:
                    self._site_begin(world)
                else:
                    self._hand_over(world)
        return Command(pos=goal, yaw=leg.yaw, arm=dict(ARM_STOW))

    def _hand_over(self, world):
        s = self.task.sites[self.k]
        p = self.params["sites"][s.key]
        self.sub = s.task.expert(p, self.arm)
        self.sub.follow = self.follow
        self.mem = s.task.reset(p, s.local(world))
        self.site_t = 0.0

    def _run_site(self, world):
        s = self.task.sites[self.k]
        p = self.params["sites"][s.key]
        local = s.local(world)
        cmd = s.to_course(self.sub(local, self.dt))
        self.detail = self.sub.state
        self.site_t += self.dt
        done = s.task.evaluate(local, p, self.mem, self.dt).success
        if done and self.k < len(self.task.sites) - 1:
            self.timeouts += [f"{s.label}:{t}" for t in self.sub.timeouts]
            self._site_begin(world)
        return cmd


class ReefTour(CourseTour):
    reef = "auto"

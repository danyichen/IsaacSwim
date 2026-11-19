from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaacswim.np_math import quat_to_rpy, rpy_to_quat, wrap_pi

__all__ = [
    "Prop", "Joint", "CurrentSpec", "SceneSpec", "World", "Command", "StepResult",
    "Task", "Expert", "yaw_quat", "heading_frame", "hold_timer",
]


@dataclass
class Prop:
    name: str
    shape: str
    size: tuple
    color: tuple = (0.5, 0.5, 0.5)
    kind: str = "static"
    density: float | None = None
    axis: str = "Z"
    friction: tuple = (1.0, 0.9)
    emissive: tuple | None = None
    reflectivity: float = 1.0
    graspable: bool = False

    def __post_init__(self):
        if self.shape not in ("box", "cylinder", "sphere"):
            raise ValueError(f"{self.name}: unknown shape {self.shape!r}")
        if self.kind not in ("static", "dynamic", "visual"):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == "dynamic" and self.density is None:
            raise ValueError(f"{self.name}: a dynamic prop needs a density")
        self.size = tuple(float(s) for s in self.size)

    @property
    def half_extents(self) -> np.ndarray:
        if self.shape == "box":
            return np.asarray(self.size) / 2.0
        if self.shape == "sphere":
            return np.full(3, self.size[0])
        r, h = self.size
        he = np.array([r, r, r])
        he["XYZ".index(self.axis)] = h / 2.0
        return he

    @property
    def volume(self) -> float:
        if self.shape == "box":
            return float(np.prod(self.size))
        if self.shape == "sphere":
            return 4.0 / 3.0 * math.pi * self.size[0] ** 3
        return math.pi * self.size[0] ** 2 * self.size[1]


@dataclass
class Joint:
    name: str
    body: str
    anchor: tuple
    axis: str = "Z"
    friction_torque: float = 0.5
    damping: float = 0.05


@dataclass
class CurrentSpec:
    mean: tuple = (0.0, 0.0, 0.0)
    sigma: tuple = (0.0, 0.0, 0.0)
    tau: float = 2.5

    @property
    def still(self) -> bool:
        return not any(self.mean) and not any(self.sigma)


@dataclass
class SceneSpec:
    seabed_z: float
    props: list[Prop]
    poses: dict[str, tuple]
    spawn_pos: tuple
    spawn_yaw: float = 0.0
    joints: list[Joint] = field(default_factory=list)
    current: CurrentSpec = field(default_factory=CurrentSpec)
    arm: dict | None = None
    camera_tilt: float = 0.0
    seabed_color: tuple = (0.28, 0.26, 0.22)

    PARK_DEPTH = 50.0

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.poses[name]
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    def park(self, name: str, slot: int = 0) -> None:
        self.poses[name] = ((100.0 + 2.0 * slot, 100.0, self.seabed_z - self.PARK_DEPTH),
                            (1.0, 0.0, 0.0, 0.0))

    def prop(self, name: str) -> Prop:
        for p in self.props:
            if p.name == name:
                return p
        raise KeyError(name)

    def validate(self) -> None:
        names = [p.name for p in self.props]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate prop names in {names}")
        missing = set(names) - set(self.poses)
        if missing:
            raise ValueError(f"no pose for {sorted(missing)}")
        for j in self.joints:
            if self.prop(j.body).kind != "dynamic":
                raise ValueError(f"joint {j.name}: {j.body} must be a dynamic prop")

    def same_layout(self, other: SceneSpec) -> bool:
        key = [(p.name, p.shape, p.size, p.kind, p.axis) for p in self.props]
        return key == [(p.name, p.shape, p.size, p.kind, p.axis) for p in other.props] \
            and [j.name for j in self.joints] == [j.name for j in other.joints]


def yaw_quat(yaw: float) -> tuple:
    return tuple(float(v) for v in rpy_to_quat((0.0, 0.0, yaw)))


def heading_frame(vec_w, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    v = np.asarray(vec_w, dtype=np.float64)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


@dataclass
class World:
    t: float
    pos: np.ndarray
    quat: np.ndarray
    lin_vel_b: np.ndarray
    ang_vel_b: np.ndarray
    arm_q: dict
    grip_q: float
    grip_closed: bool
    tool_pos: np.ndarray
    tool_R: np.ndarray
    objects: dict
    joints: dict = field(default_factory=dict)
    seabed_z: float = 0.0
    thrust_scale: float = 1.0
    grasped: str | None = None
    meter_reading: dict[str, float] | None = None

    @property
    def rpy(self) -> np.ndarray:
        return np.asarray(quat_to_rpy(self.quat))

    @property
    def yaw(self) -> float:
        return float(self.rpy[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.lin_vel_b))

    @property
    def altitude(self) -> float:
        return float(self.pos[2] - self.seabed_z)

    def obj(self, name: str) -> np.ndarray:
        return np.asarray(self.objects[name][0], dtype=np.float64)


@dataclass
class Command:
    pos: np.ndarray
    yaw: float
    arm: dict | None = None
    grip: bool = False
    meter_reading: dict[str, float] | None = None


@dataclass
class StepResult:
    reward: float
    success: bool = False
    failure: str | None = None
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.success or self.failure is not None


def hold_timer(mem: dict, key: str, ok: bool, dt: float) -> float:
    mem[key] = mem.get(key, 0.0) + dt if ok else 0.0
    return mem[key]


class Expert:
    DEADLINES: dict[str, float] = {}

    def __init__(self):
        self.state = "start"
        self.state_t = 0.0
        self.log: list[tuple[float, str]] = []
        self.timeouts: list[str] = []
        self.dt = 0.05
        self.follow = False

    def goto(self, state: str, t: float) -> None:
        self.log.append((round(t, 3), state))
        self.state, self.state_t = state, 0.0

    def expired(self) -> bool:
        lim = self.DEADLINES.get(self.state)
        if lim is not None and self.state_t > lim:
            if self.state not in self.timeouts:
                self.timeouts.append(self.state)
            return True
        return False

    def __call__(self, world: World, dt: float) -> Command:
        self.dt = dt
        if self.follow:
            self.sync(world)
        cmd = self.act(world)
        self.state_t += dt
        return cmd

    def sync(self, world: World) -> None:
        pass

    def act(self, world: World) -> Command:  # pragma: no cover
        raise NotImplementedError


class Task:
    name: str = ""
    horizon: float = 60.0
    seabed_z: float = -1.5
    templates: tuple[str, ...] = ()
    summary: str = ""
    keeper_ki: float = 25.0
    report_fields: dict[str, tuple[float, float]] = {}  # noqa: RUF012
    video_view: dict = {"eye": (-1.5, -0.8, 0.5), "look": (0.8, 0.0, -0.2),  # noqa: RUF012
                        "follow_yaw": True}

    @property
    def report_labels(self) -> tuple[str, ...]:
        return ("report", *self.report_fields) if self.report_fields else ()

    def encode_report(self, reading: dict[str, float] | None) -> np.ndarray:
        out = -np.ones(len(self.report_labels))
        if reading is not None and out.size:
            out[0] = 1.0
            for i, (name, (lo, hi)) in enumerate(self.report_fields.items(), start=1):
                out[i] = 2.0 * (reading[name] - lo) / (hi - lo) - 1.0
        return out

    def decode_report(self, values) -> dict[str, float] | None:
        values = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
        if not values.size or values[0] <= 0.0:
            return None
        return {name: lo + 0.5 * (v + 1.0) * (hi - lo)
                for v, (name, (lo, hi)) in zip(values[1:], self.report_fields.items(), strict=True)}

    def dress(self, stage, *, seed: int = 0, root: str = "/World",
              origin=(0.0, 0.0, 0.0)) -> tuple[str, ...]:
        return ()

    def label_sonar(self, osim, stage, roots: tuple[str, ...]) -> None:
        pass

    def sample(self, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def scene(self, params: dict) -> SceneSpec:
        raise NotImplementedError

    def instruction(self, params: dict, rng: np.random.Generator | None = None) -> str:
        t = self.templates[0] if rng is None else self.templates[
            int(rng.integers(len(self.templates)))]
        return t.format(**params)

    def reset(self, params: dict, world: World) -> dict:
        return {}

    def evaluate(self, world: World, params: dict, mem: dict, dt: float) -> StepResult:
        raise NotImplementedError

    def privileged(self, world: World, params: dict) -> np.ndarray:
        raise NotImplementedError

    def expert(self, params: dict, arm) -> Expert:
        raise NotImplementedError

    def metrics(self, world: World, params: dict, mem: dict) -> dict[str, Any]:
        return {}

    def current(self, world: World, params: dict, mem: dict) -> CurrentSpec | None:
        return None

    def keeper_gain(self, world: World, params: dict, mem: dict) -> float | None:
        return None

    @staticmethod
    def yaw_error(world: World, yaw: float) -> float:
        return abs(wrap_pi(yaw - world.yaw))

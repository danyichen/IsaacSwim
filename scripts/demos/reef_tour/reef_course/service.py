import math
from dataclasses import replace

import numpy as np

from isaacswim.np_math import wrap_pi
from isaacswim_bench.core import Command, Expert, Prop, StepResult
from isaacswim_bench.platform import ARM_STOW
from isaacswim_bench.tasks.pipeline import ALTITUDE, PipelineInspect
from isaacswim_bench.tasks.sample import SampleCollect, _PickPlace
from isaacswim_bench.tasks.station_keep import HOLD_S, POS_TOL, YAW_TOL, StationKeep, _GoAndHold
from isaacswim_bench.tasks.valve import ValveTurn, _TurnValve

SERVICE_PIPE = [
    (-10.0, -3.0),
    (-8.5, -3.0),
    (-7.066995, -2.556720),
    (-5.574489, -2.406970),
    (-4.141484, -2.850250),
    (-2.4, -3.05),
    (-1.0, -2.95),
    (0.25, -2.10),
    (0.25, -1.0),
    (0.75, -1.0),
]


class ConnectedPipeline(PipelineInspect):
    horizon = 150.0
    summary = "inspect the circulation line all the way to its control valve"
    templates = (
        "Inspect the circulation pipeline to its valve manifold, "
        "staying about one metre above the line.",
    )

    def vertices(self, params):
        return np.asarray(params.get("vertices", SERVICE_PIPE), dtype=float)

    def lengths(self, params):
        return np.linalg.norm(np.diff(self.vertices(params), axis=0), axis=1)

    def point_at(self, params, s):
        v = self.vertices(params)
        lengths = self.lengths(params)
        ends = np.cumsum(lengths)
        s = float(np.clip(s, 0, ends[-1]))
        k = min(int(np.searchsorted(ends, s, side="right")), len(lengths) - 1)
        u = (s - (ends[k - 1] if k else 0)) / lengths[k]
        d = v[k + 1] - v[k]
        return v[k] + u * d, math.atan2(d[1], d[0])

    def project(self, params, xy):
        best = (0.0, math.inf)
        arc = 0.0
        for a, b, L in zip(
            self.vertices(params)[:-1], self.vertices(params)[1:], self.lengths(params)
        ):
            d = b - a
            u = float(np.clip(np.dot(np.asarray(xy) - a, d) / (L * L), 0, 1))
            dist = float(np.linalg.norm(np.asarray(xy) - a - u * d))
            if dist < best[1]:
                best = (arc + u * L, dist)
            arc += L
        return best

    def metrics(self, world, params, mem):
        return {
            "coverage": float(mem["seen"].mean()),
            "path_m": mem["path"],
            "pipe_m": float(self.lengths(params).sum()),
        }

    def privileged(self, world, params):
        result = super().privileged(world, params)
        s, _ = self.project(params, world.pos[:2])
        result[-2] = self.lengths(params).sum() - s
        return result

    def expert(self, params, arm):
        return _ServicePipeExpert(self, params)


class _ServicePipeExpert(Expert):
    def __init__(self, task, params):
        super().__init__()
        self.task = task
        self.params = params
        self.s = 0.0
        self.state = "acquire"

    def act(self, world):
        p, h = self.task.point_at(self.params, self.s)
        z = self.task.seabed_z + ALTITUDE
        if self.state == "acquire":
            d = np.linalg.norm(world.pos[:2] - p)
            if d < 0.2 and abs(world.pos[2] - z) < 0.15:
                self.goto("follow", world.t)
            return Command(
                np.array([*p, z]),
                math.atan2(p[1] - world.pos[1], p[0] - world.pos[0]) if d > 0.6 else h,
                dict(ARM_STOW),
            )
        s, _ = self.task.project(self.params, world.pos[:2])
        L = self.task.lengths(self.params).sum()
        self.s = min(L, max(self.s, min(s + 0.30, self.s + 0.18 * self.dt)))
        p, h = self.task.point_at(self.params, self.s)
        if self.s >= L:
            self.state = "arrive"
        return Command(np.array([*p, z]), h, dict(ARM_STOW))


class _QuickValve(_TurnValve):
    GOTO_TOL = 0.02
    GOTO_SPEED = 0.06


class TourValve(ValveTurn):
    def expert(self, params, arm):
        return _QuickValve(self, params, arm)


PLUG_R = 0.0216
PLUG_H = 0.0412
PLUG_GRASP_DEPTH = 0.0165
STATION_CURRENT = 0.2
STATION_GUST = 0.06


class CoralRelocate(SampleCollect):
    name = "coral_relocate"
    sample_height = PLUG_H
    sample_width = 2.0 * PLUG_R
    grasp_depth = PLUG_GRASP_DEPTH
    summary = "move the tagged loose coral fragment into its nursery holder"
    templates = (
        "Move the coral fragment with the {color} identification collar into the nursery holder.",
        "Pick up the loose coral fragment tagged {color} and place it in the nursery holder.",
    )

    def sample(self, rng):
        p = super().sample(rng)
        p["basket"] = [1.6, 3.7]
        p["basket_yaw"] = 0.0
        return p

    def scene(self, params):
        spec = super().scene(params)
        poses = dict(spec.poses)

        def body(p):
            if p.kind != "dynamic" or not p.name.startswith("sample_"):
                return p
            x, y, _ = poses[p.name][0]
            poses[p.name] = ((x, y, self.seabed_z + PLUG_H / 2 + 0.001), poses[p.name][1])
            return replace(p, shape="cylinder", size=(PLUG_R, PLUG_H), axis="Z", density=1800.0)

        props = [body(p) for p in spec.props]
        return replace(spec, props=props, poses=poses)

    def expert(self, params, arm):
        return _CarryExpert(self, params, arm)


class _CarryExpert(_PickPlace):
    TURN_RANGE = 1.0
    TURN_TOL = math.radians(10.0)
    ARRIVE = 0.15
    DEADLINES = {**_PickPlace.DEADLINES, "transit": 60.0}
    turn_xy = None
    FACE = math.pi

    def act(self, world):
        if self.state == "settle":
            self.hold_yaw = self.FACE
            if self.arm_cmd is None:
                self.arm_cmd = dict(world.arm_q)
            if (
                self.state_t > 0.5 and abs(wrap_pi(world.yaw - self.FACE)) < 0.05
            ) or self.expired():
                y = self.params["sample_yaws"][self.params["target"]]
                self.arm_cmd = self.plan_grasp_arm(world, (-math.sin(y), math.cos(y), 0.0))
                self.goto("goto", world.t)
            return self.cmd_hold(world, False)
        if self.state != "transit":
            self.turn_xy = None
        else:
            d = self.drop_xy - world.tool_pos[:2]
            dist = float(np.linalg.norm(d))
            if dist > self.ARRIVE:
                self.hold_yaw = math.atan2(d[1], d[0])
                if (
                    dist > self.TURN_RANGE
                    and abs(wrap_pi(self.hold_yaw - world.yaw)) > self.TURN_TOL
                ):
                    if self.turn_xy is None:
                        self.turn_xy = world.tool_pos[:2].copy()
                    return self.cmd_tool(world, np.array([*self.turn_xy, self.carry_z]), True)
        return super().act(world)


GAUGES = {
    "pressure_bar": dict(lo=0.0, hi=10.0, y=0.22, z=0.2, tol=0.15, values=(2.0, 8.0), step=0.1),
    "temperature_c": dict(lo=0.0, hi=40.0, y=-0.22, z=0.2, tol=0.5, values=(24.0, 30.0), step=0.5),
    "ph": dict(lo=6.0, hi=10.0, y=0.22, z=-0.2, tol=0.15, values=(7.8, 8.4), step=0.1),
    "flow_ms": dict(lo=0.0, hi=1.0, y=-0.22, z=-0.2, tol=0.05, values=(0.05, 0.40), step=0.05),
}


class MeterRead(StationKeep):
    name = "meter_read"
    horizon = 90.0
    report_fields = {k: (g["lo"], g["hi"]) for k, g in GAUGES.items()}  # noqa: RUF012
    summary = (
        "hold steady at the instrument station, "
        "then report its pressure, temperature, pH and current speed"
    )
    templates = (
        "Hold one metre off the instrument station for ten seconds, then report its pressure "
        "in bar, temperature in degrees Celsius, pH and current speed in metres per second.",
    )

    def sample(self, rng):
        p = super().sample(rng)
        for k, g in GAUGES.items():
            n = round((g["values"][1] - g["values"][0]) / g["step"])
            p[k] = round(g["values"][0] + g["step"] * int(rng.integers(0, n + 1)), 3)
        current = np.asarray(p["current"], dtype=np.float64)
        speed = float(np.linalg.norm(current))
        p["current"] = (current * min(1.0, STATION_CURRENT / max(speed, 1e-9))).tolist()
        p["gust"] = min(float(p["gust"]), STATION_GUST)
        return p

    def scene(self, params):
        spec = super().scene(params)
        yaw = params["panel_yaw"]
        n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        t = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        props = []
        poses = {}
        for k, g in GAUGES.items():
            a = math.radians(225 - 270 * (params[k] - g["lo"]) / (g["hi"] - g["lo"]))
            ca, sa = math.cos(a / 2), math.sin(a / 2)
            center = (
                np.array([*params["panel_xy"], self.seabed_z + 1.0 + g["z"]])
                + g["y"] * t
                + 0.087 * n
            )
            name = "meter_needle_" + k
            props.append(
                Prop(name, "box", (0.008, 0.088, 0.005), color=(0.055, 0.045, 0.035), kind="visual")
            )
            poses[name] = (
                tuple(center + 0.044 * (math.cos(a) * t + math.sin(a) * up)),
                (cy * ca, cy * sa, sy * sa, sy * ca),
            )
        return replace(spec, props=props, poses=poses)

    def reset(self, params, world):
        mem = super().reset(params, world)
        mem.update(reading=None, read=False)
        return mem

    def evaluate(self, world, params, mem, dt):
        held = super().evaluate(world, params, mem, dt)
        reading = world.meter_reading
        if held.success and reading is not None:
            mem["reading"] = dict(reading)
            mem["read"] = all(
                k in reading and abs(reading[k] - params[k]) <= g["tol"] for k, g in GAUGES.items()
            )
            if not mem["read"]:
                return StepResult(
                    -1.0,
                    failure="incorrect_meter_reading",
                    info={**held.info, "reported": dict(reading)},
                )
        return StepResult(
            held.reward + (1.0 if mem["read"] else 0),
            success=bool(held.success and mem["read"]),
            info={**held.info, "reading_submitted": mem["reading"] is not None},
        )

    def metrics(self, world, params, mem):
        return {
            **super().metrics(world, params, mem),
            "reported": mem["reading"],
            "expected": {k: params[k] for k in GAUGES},
            "reading_correct": mem["read"],
        }

    def expert(self, params, arm):
        return _ReadMeterExpert(self, params)


class _ReadMeterExpert(_GoAndHold):
    def __init__(self, task, params):
        super().__init__(*task.target(params))
        self.task = task
        self.params = params
        self.held = 0.0

    def act(self, world):
        cmd = super().act(world)
        ok = (
            np.linalg.norm(world.pos - self.pos) < POS_TOL
            and self.task.yaw_error(world, self.yaw) < YAW_TOL
        )
        self.held = self.held + self.dt if ok else 0.0
        if self.held >= HOLD_S + 0.15:
            self.state = "report"
            cmd.meter_reading = {k: self.params[k] for k in GAUGES}
        return cmd

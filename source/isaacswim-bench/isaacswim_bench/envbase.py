from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from isaacswim import IMU, Altimeter, DopplerVelocityLog, PressureDepth
from isaacswim.control import VehicleState
from isaacswim.np_math import wrap_pi

from .actions import PilotAction
from .core import Command, CurrentSpec, SceneSpec, StepResult, Task, World
from .platform import ARM_JOINTS

__all__ = ["BenchEnv", "SURFACE_Z", "PROPRIO_KEYS", "CORE_REPORT_KEYS"]

SURFACE_Z = 10.0
PROPRIO_KEYS = ("depth", "altitude", "imu", "dvl", "arm_q", "gripper")
CORE_REPORT_KEYS = ("task", "seed", "instruction", "success", "failure", "sim_seconds",
                    "steps", "roll_max_deg", "pitch_max_deg", "scale_min",
                    "saturated_fraction", "path_m", "expert_timeouts")


class BenchEnv:
    def __init__(self, task: Task, *, arm, control_hz: float = 20.0,
                 action_mode: str = "pilot", sensor_noise: float = 1.0,
                 paraphrase: bool = True, current_scale: float = 1.0):
        self.task = task
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if not math.isfinite(sensor_noise) or sensor_noise < 0:
            raise ValueError("sensor_noise must be finite and nonnegative")
        if not math.isfinite(current_scale) or current_scale < 0:
            raise ValueError("current_scale must be finite and nonnegative")
        self.current_scale = float(current_scale)
        self.arm = arm
        self.control_dt = 1.0 / float(control_hz)
        self.actions = PilotAction(self.control_dt, limits=arm.limits, mode=action_mode)
        self.paraphrase = paraphrase
        k = float(sensor_noise)
        self.sensors = {
            "depth": PressureDepth(surface_height=SURFACE_Z, sigma=0.01 * k),
            "altitude": Altimeter(min_range=0.1, sigma=0.01 * k),
            "imu": IMU(attitude_sigma=0.005 * k, gyro_sigma=0.002 * k),
            "dvl": DopplerVelocityLog(sigma=0.003 * k, scale_error=0.0),
        }
        self._gen = torch.Generator().manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.world: World | None = None
        self.params: dict = {}
        self.spec: SceneSpec | None = None
        self.instruction = ""
        self._expert = None
        self.expert_follows = False
        self._mem: dict = {}

    def _reset_plant(self, spec: SceneSpec) -> World:
        raise NotImplementedError

    def _advance(self, setpoints, dt: float) -> World:
        raise NotImplementedError

    def _set_current(self, current: CurrentSpec) -> None:
        pass

    def _set_keeper_ki(self, ki: float) -> None:
        pass

    def _extra_obs(self) -> dict[str, Any]:
        return {}

    def _nav_pose(self) -> tuple[np.ndarray, float]:
        return self.world.pos, self.world.yaw

    def _nav_heading_error(self) -> float:
        return 0.0

    def to_nav(self, cmd: Command) -> Command:
        pos_n, _ = self._nav_pose()
        b = self._nav_heading_error()
        d = np.asarray(cmd.pos, dtype=np.float64) - self.world.pos
        c, s = math.cos(b), math.sin(b)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        return Command(pos=np.asarray(pos_n, dtype=np.float64) + d,
                       yaw=wrap_pi(cmd.yaw + b), arm=cmd.arm, grip=cmd.grip,
                       meter_reading=cmd.meter_reading)

    @property
    def action_size(self) -> int:
        return self.actions.size + len(self.task.report_labels)

    @property
    def action_labels(self) -> tuple[str, ...]:
        return (*self.actions.labels, *self.task.report_labels)

    @property
    def expert(self):
        return self._expert

    @property
    def stage(self) -> int:
        return int(self._mem.get("next", 0))

    def zero_action(self) -> np.ndarray:
        action = np.zeros(self.action_size)
        action[self.actions.size - 1] = -1.0
        action[self.actions.size:] = self.task.encode_report(None)
        return action

    def split_action(self, action) -> tuple[np.ndarray, dict[str, float] | None]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_size:
            raise ValueError(f"action has {a.shape[0]} numbers, expected {self.action_size} "
                             f"({', '.join(self.action_labels)})")
        if not np.isfinite(a).all():
            raise ValueError("action contains NaN or infinity")
        n = self.actions.size
        return a[:n], self.task.decode_report(a[n:])

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._gen.manual_seed(int(seed))
        self.seed = seed
        self.params = (options or {}).get("params") or self.task.sample(self.rng)
        self.spec = self.task.scene(self.params)
        if self.current_scale != 1.0:
            c, k = self.spec.current, self.current_scale
            self.spec.current = CurrentSpec(mean=tuple(k * v for v in c.mean),
                                            sigma=tuple(k * v for v in c.sigma), tau=c.tau)
        self.spec.validate()
        self.instruction = self.task.instruction(
            self.params, self.rng if self.paraphrase else None)
        self.world = self._reset_plant(self.spec)
        self.actions.reset(*self._nav_pose(), self.world.arm_q, grip=False)
        self._mem = self.task.reset(self.params, self.world)
        self._expert = None
        self._last: StepResult | None = None
        self.stats = _Stats(self.world)
        return self.observe(), {"instruction": self.instruction, "params": self.params}

    def step(self, action):
        control, reading = self.split_action(action)
        pos_n, yaw_n = self._nav_pose()
        sp = self.actions.apply(control, pos=pos_n, yaw=yaw_n)
        self.world = replace(self._advance(sp, self.control_dt), meter_reading=reading)
        return self._finish_step(sp)

    def _finish_step(self, sp):
        if not all(np.isfinite(v).all() for v in (
                self.world.pos, self.world.quat, self.world.lin_vel_b,
                self.world.ang_vel_b, self.world.tool_pos)):
            raise FloatingPointError(
                f"{self.task.name}: nonfinite physics state at t={self.world.t:.2f}s")
        res = self.task.evaluate(self.world, self.params, self._mem, self.control_dt)
        self._last = res
        cur = self.task.current(self.world, self.params, self._mem)
        if cur is not None:
            k = self.current_scale
            self._set_current(CurrentSpec(mean=tuple(k * v for v in cur.mean),
                                          sigma=tuple(k * v for v in cur.sigma), tau=cur.tau))
        ki = self.task.keeper_gain(self.world, self.params, self._mem)
        if ki is not None:
            self._set_keeper_ki(ki)
        self.stats.update(self.world, sp)
        truncated = self.world.t >= self.task.horizon - 1e-9 and not res.done
        info = {"success": res.success, "failure": res.failure, **res.info}
        return self.observe(), float(res.reward), res.done, truncated, info

    def expert_command(self) -> Command:
        if self._expert is None:
            self._expert = self.task.expert(self.params, self.arm)
            self._expert.follow = self.expert_follows
        return self._expert(self.world, self.control_dt)

    def expert_action(self) -> np.ndarray:
        cmd = self.to_nav(self.expert_command())
        control = self.actions.encode(cmd)
        return np.concatenate([control, self.task.encode_report(cmd.meter_reading)])

    def vehicle_state(self) -> VehicleState:
        w = self.world
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32)).unsqueeze(0)  # noqa: E731
        return VehicleState(pos_w=t(w.pos), quat_w=t(w.quat), lin_vel_b=t(w.lin_vel_b),
                            ang_vel_b=t(w.ang_vel_b),
                            joint_pos=t([w.arm_q[n] for n in ARM_JOINTS]),
                            joint_vel=t(np.zeros(len(ARM_JOINTS))),
                            joint_names=ARM_JOINTS, time=w.t,
                            altitude=t([w.altitude])[0])

    def observe(self) -> dict[str, Any]:
        w = self.world
        state = self.vehicle_state()
        obs: dict[str, Any] = {}
        for name, sensor in self.sensors.items():
            out = sensor.measure(state, generator=self._gen)
            if isinstance(out, tuple):
                out = out[0]
            obs[name] = out[0].reshape(-1).numpy().astype(np.float32)
        obs["arm_q"] = np.array([w.arm_q[n] for n in ARM_JOINTS], dtype=np.float32)
        obs["gripper"] = np.array([w.grip_q, float(w.grip_closed)], dtype=np.float32)
        obs["proprio"] = np.concatenate([obs[k] for k in PROPRIO_KEYS])
        obs["priv_task"] = self.task.privileged(w, self.params)
        obs["priv_odom"] = np.concatenate([w.pos, w.quat, w.lin_vel_b,
                                           w.ang_vel_b]).astype(np.float32)
        obs.update(self._extra_obs())
        obs["instruction"] = self.instruction
        return obs

    def report(self) -> dict[str, Any]:
        res = self._last
        s = self.stats
        out = {
            "task": self.task.name, "seed": self.seed, "instruction": self.instruction,
            "success": bool(res and res.success),
            "failure": (res.failure if res else None) or (
                None if res and res.success else "timeout"),
            "sim_seconds": round(self.world.t, 3), "steps": s.steps,
            "roll_max_deg": s.roll_max, "pitch_max_deg": s.pitch_max,
            "scale_min": s.scale_min,
            "saturated_fraction": s.saturated / max(s.steps, 1),
            "path_m": s.path,
            "expert_timeouts": list(self._expert.timeouts) if self._expert else None,
        }
        out.update(self.task.metrics(self.world, self.params, self._mem))
        return out


class _Stats:
    def __init__(self, world: World):
        self.steps = 0
        self.roll_max = self.pitch_max = 0.0
        self.scale_min = 1.0
        self.saturated = 0
        self.path = 0.0
        self._last = world.pos.copy()

    def update(self, w: World, sp) -> None:
        self.steps += 1
        r, p, _ = w.rpy
        self.roll_max = max(self.roll_max, abs(math.degrees(r)))
        self.pitch_max = max(self.pitch_max, abs(math.degrees(p)))
        self.scale_min = min(self.scale_min, float(w.thrust_scale))
        self.saturated += int(w.thrust_scale < 1.0 - 1e-6)
        self.path += float(np.linalg.norm(w.pos - self._last))
        self._last = w.pos.copy()

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from isaacswim.np_math import wrap_pi

from .core import Command
from .platform import ARM_JOINTS, JOINT_RATE

__all__ = ["PilotAction", "Setpoints", "ACTION_LABELS", "WRENCH_SCALE"]

ACTION_LABELS = ("surge", "sway", "heave", "yaw_rate",
                 "qd_e", "qd_d", "qd_c", "qd_b", "grip")

WRENCH_SCALE = np.array([80.0, 80.0, 80.0, 12.0, 12.0, 12.0])


@dataclass
class Setpoints:
    pos: np.ndarray
    yaw: float
    arm: dict
    grip: bool = False
    wrench: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


class PilotAction:
    def __init__(self, dt: float, *, v_max=(0.30, 0.30, 0.20, 0.50),
                 qd_max: float = JOINT_RATE, leash: float = 0.25,
                 yaw_leash: float = 0.35, limits: dict | None = None,
                 mode: str = "pilot"):
        if mode not in ("pilot", "wrench"):
            raise ValueError(f"mode must be 'pilot' or 'wrench', not {mode!r}")
        self.dt = float(dt)
        self.v_max = np.asarray(v_max, dtype=np.float64)
        self.qd_max = float(qd_max)
        self.leash = float(leash)
        self.yaw_leash = float(yaw_leash)
        self.limits = dict(limits or {})
        self.mode = mode
        self.sp: Setpoints | None = None

    @property
    def size(self) -> int:
        return 9 if self.mode == "pilot" else 11

    @property
    def labels(self) -> tuple[str, ...]:
        if self.mode == "pilot":
            return ACTION_LABELS
        return ("fx", "fy", "fz", "mx", "my", "mz") + ACTION_LABELS[4:]

    def reset(self, pos, yaw: float, arm_q: dict, grip: bool = False) -> Setpoints:
        self.sp = Setpoints(pos=np.asarray(pos, dtype=np.float64).copy(), yaw=float(yaw),
                            arm={n: float(arm_q[n]) for n in ARM_JOINTS}, grip=bool(grip))
        return self.sp

    def apply(self, action, *, pos=None, yaw: float | None = None) -> Setpoints:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.size:
            raise ValueError(f"action has {a.shape[0]} numbers, expected {self.size} "
                             f"({', '.join(self.labels)})")
        if not np.isfinite(a).all():
            raise ValueError("action contains NaN or infinity")
        a = np.clip(a, -1.0, 1.0)
        sp = self.sp
        if self.mode == "pilot":
            v = a[:4] * self.v_max * self.dt
            c, s = math.cos(sp.yaw), math.sin(sp.yaw)
            sp.pos = sp.pos + np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
            sp.yaw = wrap_pi(sp.yaw + v[3])
            if pos is not None:
                pos = np.asarray(pos, dtype=np.float64)
                sp.pos = pos + np.clip(sp.pos - pos, -self.leash, self.leash)
            if yaw is not None:
                err = wrap_pi(sp.yaw - yaw)
                sp.yaw = wrap_pi(yaw + max(-self.yaw_leash, min(self.yaw_leash, err)))
            arm_a = a[4:8]
        else:
            sp.wrench = a[:6] * WRENCH_SCALE
            arm_a = a[6:10]
        for n, qd in zip(ARM_JOINTS, arm_a, strict=True):
            q = sp.arm[n] + qd * self.qd_max * self.dt
            lo, hi = self.limits.get(n, (-math.inf, math.inf))
            sp.arm[n] = min(max(q, lo), hi)
        sp.grip = bool(a[-1] > 0.0)
        return sp

    def encode(self, cmd: Command) -> np.ndarray:
        if self.mode != "pilot":
            raise ValueError("encode() needs mode='pilot'")
        sp = self.sp
        a = np.zeros(self.size)
        d = np.asarray(cmd.pos, dtype=np.float64) - sp.pos
        c, s = math.cos(sp.yaw), math.sin(sp.yaw)
        d_h = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
        a[:3] = d_h / (self.v_max[:3] * self.dt)
        a[3] = wrap_pi(cmd.yaw - sp.yaw) / (self.v_max[3] * self.dt)
        if cmd.arm is not None:
            for k, n in enumerate(ARM_JOINTS):
                if n in cmd.arm:
                    a[4 + k] = (cmd.arm[n] - sp.arm[n]) / (self.qd_max * self.dt)
        a[8] = 1.0 if cmd.grip else -1.0
        m = np.max(np.abs(a[:3]))
        if m > 1.0:
            a[:3] /= m
        return np.clip(a, -1.0, 1.0)

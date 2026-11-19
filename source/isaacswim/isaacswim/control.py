"""The vehicle state and the sensor and controller interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

__all__ = ["VehicleState", "Controller", "Sensor"]


@dataclass
class VehicleState:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_b: torch.Tensor
    ang_vel_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_names: Sequence[str] = ()
    body_names: Sequence[str] = ()
    time: float = 0.0
    altitude: torch.Tensor | None = None
    current_b: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return int(self.pos_w.shape[0])

    def env(self, i: int = 0) -> dict[str, Any]:
        def np1(t):
            return None if t is None else t[i].detach().cpu().numpy().astype(np.float64)

        jp = np1(self.joint_pos)
        return {
            "pos_w": np1(self.pos_w),
            "quat_w": np1(self.quat_w),
            "lin_vel_b": np1(self.lin_vel_b),
            "ang_vel_b": np1(self.ang_vel_b),
            "joint_pos": {n: float(v) for n, v in zip(self.joint_names, jp)},
            "altitude": None if self.altitude is None else float(self.altitude[i]),
            "current_b": np1(self.current_b),
            "time": self.time,
        }


@runtime_checkable
class Sensor(Protocol):
    def measure(self, state: VehicleState, *, generator=None): ...


@runtime_checkable
class Controller(Protocol):
    def __call__(self, obs: Mapping[str, Any], dt: float) -> Mapping[str, Any]: ...

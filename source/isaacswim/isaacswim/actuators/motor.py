from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["Motor"]

torque_like = torch.Tensor | float


@dataclass
class Motor:
    ke: float
    resistance: float
    inertia: float
    voltage: float

    @classmethod
    def t200(cls, voltage: float = 16.0, **kw) -> Motor:
        args = dict(ke=0.01289, resistance=0.4705, inertia=1.0e-5, voltage=voltage)
        args.update(kw)
        return cls(**args)

    def duty_for_speed(self, w: torque_like, q: torque_like) -> torch.Tensor:
        return (self.ke * w + self.resistance * q / self.ke) / self.voltage

    def bollard_speed(self, duty: float, c: float) -> float:
        a, b = c, self.ke ** 2 / self.resistance
        drive = self.ke * abs(duty) * self.voltage / self.resistance
        w = (-b + math.sqrt(b * b + 4 * a * drive)) / (2 * a)
        return math.copysign(w, duty)

    def step(self, w: torch.Tensor, duty: torch.Tensor, load, dt: float,
             iters: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = w
        for _ in range(iters):
            q, dq = load(w)
            f = (self.ke * (duty * self.voltage - self.ke * w) / self.resistance - q) / self.inertia
            df = (-self.ke ** 2 / self.resistance - dq) / self.inertia
            w = w - (w - w0 - dt * f) / (1.0 - dt * df)
        return w, (duty * self.voltage - self.ke * w) / self.resistance

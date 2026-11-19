from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Propeller", "advance_speed"]

_T200_V = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
_T200_RPM_FWD = (2668.0, 2995.0, 3288.0, 3533.0, 3751.0, 3931.0)
_T200_RPM_REV = (2667.0, 2976.0, 3244.0, 3468.0, 3643.0, 3783.0)


@dataclass
class Propeller:
    diameter: float
    kt0_fwd: float
    kt0_rev: float
    j0_fwd: float
    j0_rev: float
    max_rps_fwd: float
    max_rps_rev: float
    wake_fraction: float = 0.0
    kt_curve: tuple | None = None
    rho_ref: float = 1000.0
    kq0_fwd: float | None = None
    kq0_rev: float | None = None

    @classmethod
    def t200(cls, voltage: float = 16.0, **kw) -> Propeller:
        if not _T200_V[0] <= voltage <= _T200_V[-1]:
            raise ValueError(f"T200 data covers {_T200_V[0]}-{_T200_V[-1]} V, "
                             f"not {voltage}")
        args = dict(
            diameter=0.076, kt0_fwd=0.441, kt0_rev=0.351, j0_fwd=0.97, j0_rev=0.97,
            max_rps_fwd=float(np.interp(voltage, _T200_V, _T200_RPM_FWD)) / 60.0,
            max_rps_rev=float(np.interp(voltage, _T200_V, _T200_RPM_REV)) / 60.0,
            rho_ref=1000.0, kq0_fwd=0.0371, kq0_rev=0.0380,
        )
        args.update(kw)
        return cls(**args)

    def bollard_thrust(self, n, rho: float):
        kt0 = _pick(n >= 0, self.kt0_fwd, self.kt0_rev, n) if torch.is_tensor(n) \
            else (self.kt0_fwd if n >= 0 else self.kt0_rev)
        return rho * self.diameter ** 4 * kt0 * n * abs(n)

    def max_thrust(self, rho: float) -> tuple[float, float]:
        D4 = rho * self.diameter ** 4
        return (D4 * self.kt0_fwd * self.max_rps_fwd ** 2,
                D4 * self.kt0_rev * self.max_rps_rev ** 2)

    def rps_for_thrust(self, thrust: torch.Tensor, rho: float,
                       clamp: bool = True) -> torch.Tensor:
        D4 = rho * self.diameter ** 4
        fwd = thrust >= 0
        kt0 = _pick(fwd, self.kt0_fwd, self.kt0_rev, thrust)
        n = torch.sign(thrust) * torch.sqrt(thrust.abs() / (D4 * kt0))
        return n.clamp(-self.max_rps_rev, self.max_rps_fwd) if clamp else n

    def thrust(self, n: torch.Tensor, u: torch.Tensor, rho: float) -> torch.Tensor:
        D = self.diameter
        ua = (1.0 - self.wake_fraction) * u
        fwd = n >= 0
        kt0 = _pick(fwd, self.kt0_fwd, self.kt0_rev, n)
        j0 = _pick(fwd, self.j0_fwd, self.j0_rev, n)
        an = n.abs()
        T = rho * D ** 3 * an * (kt0 * D * n - (kt0 / j0) * ua)
        if self.kt_curve is None:
            return T
        Js, KTs = (torch.as_tensor(np.asarray(c, dtype=np.float64), dtype=n.dtype,
                                   device=n.device) for c in self.kt_curve)
        J = ua / (an * D).clamp(min=1e-9)
        use = fwd & (J >= 0) & (an > 1e-6)
        kt = _interp(J, Js, KTs)
        return torch.where(use, rho * D ** 4 * kt * n * n, T)

    def torque(self, n: torch.Tensor, u: torch.Tensor,
               rho: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kq0_fwd is None or self.kq0_rev is None:
            raise ValueError("this Propeller has no torque coefficients (kq0_fwd, kq0_rev)")
        D = self.diameter
        ua = (1.0 - self.wake_fraction) * u
        fwd = n >= 0
        kq0 = _pick(fwd, self.kq0_fwd, self.kq0_rev, n)
        j0 = _pick(fwd, self.j0_fwd, self.j0_rev, n)
        an = n.abs()
        q = rho * D ** 4 * an * (kq0 * D * n - (kq0 / j0) * ua)
        dq_dn = rho * D ** 4 * (2.0 * kq0 * D * an - torch.sign(n) * (kq0 / j0) * ua)
        return q, dq_dn / (2.0 * math.pi)

    def __repr__(self) -> str:
        return (f"Propeller(D={self.diameter:g} m, K_T0={self.kt0_fwd:g}/"
                f"{self.kt0_rev:g}, J0={self.j0_fwd:g}/{self.j0_rev:g}, "
                f"max {self.max_rps_fwd * 60:.0f}/{self.max_rps_rev * 60:.0f} rpm"
                f"{', measured K_T(J)' if self.kt_curve is not None else ''})")


def _pick(cond: torch.Tensor, a: float, b: float, like: torch.Tensor) -> torch.Tensor:
    c = cond.to(like.dtype)
    return a * c + b * (1.0 - c)


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]
    w = (x - x0) / (x1 - x0)
    return fp[i - 1] + w * (fp[i] - fp[i - 1])


def advance_speed(lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor,
                  current_b: torch.Tensor | None, pos: torch.Tensor,
                  push: torch.Tensor) -> torch.Tensor:
    v = lin_vel_b.unsqueeze(1) + torch.linalg.cross(
        ang_vel_b.unsqueeze(1).expand(-1, pos.shape[0], -1),
        pos.unsqueeze(0).expand(lin_vel_b.shape[0], -1, -1), dim=-1)
    if current_b is not None:
        v = v - current_b.unsqueeze(1)
    return (v * push.unsqueeze(0)).sum(-1)


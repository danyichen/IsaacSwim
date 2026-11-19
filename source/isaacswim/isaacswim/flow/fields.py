from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import torch

__all__ = [
    "VelocityField",
    "Uniform",
    "GustyCurrent",
    "Jet",
    "Pipe",
    "BoundaryLayer",
    "Composite",
    "shear_profile",
]


@runtime_checkable
class VelocityField(Protocol):
    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        ...

    def step(self, dt: float) -> None:
        pass


def shear_profile(pos: torch.Tensor, seabed_z: float, thickness: float,
                  exponent: float = 1.0 / 7.0) -> torch.Tensor:
    height = (pos[..., 2] - seabed_z).clamp(min=0.0)
    return (height / thickness).clamp(0.0, 1.0).pow(exponent).unsqueeze(-1)


class Uniform:
    def __init__(self, velocity=(0.0, 0.0, 0.0), *, seabed_z: float | None = None,
                 layer: float = 1.5, device: str = "cpu"):
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.seabed_z = seabed_z
        self.layer = float(layer)
        self._vec = torch.as_tensor(self.velocity, device=device,
                                    dtype=torch.float32)

    def step(self, dt: float) -> None:
        pass

    @property
    def stats(self) -> dict:
        return {"mean": self.velocity.tolist(),
                "mean_speed": float(np.linalg.norm(self.velocity))}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        v = self._vec.to(pos.device).expand_as(pos)
        if self.seabed_z is None:
            return v.clone()
        return v * shear_profile(pos, self.seabed_z, self.layer)


class GustyCurrent:
    def __init__(self, *, mean=(0.10, 0.03, 0.0), sigma=(0.07, 0.07, 0.03),
                 tau: float = 2.5, seabed_z: float | None = -0.55,
                 layer: float = 1.5, device: str = "cpu", seed: int = 0):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.tau = float(tau)
        self.seabed_z = seabed_z
        self.layer = float(layer)
        self.device = device
        self._rng = np.random.default_rng(seed)
        self.gust = np.zeros(3)
        self._host = torch.zeros(3, dtype=torch.float32)
        self._vec = torch.zeros(3, device=device, dtype=torch.float32)
        self._peak = 0.0
        self._sum = 0.0
        self._n = 0
        self._refresh()

    def _refresh(self) -> None:
        self._host.copy_(torch.from_numpy(self.mean + self.gust).float())
        self._vec.copy_(self._host)

    def step(self, dt: float) -> None:
        a = np.exp(-dt / self.tau)
        noise = self._rng.standard_normal(3)
        self.gust = a * self.gust + self.sigma * np.sqrt(1.0 - a * a) * noise
        self._refresh()
        speed = float(np.linalg.norm(self.mean + self.gust))
        self._peak = max(self._peak, speed)
        self._sum += speed
        self._n += 1

    @property
    def stats(self) -> dict:
        return {
            "mean_speed": self._sum / max(self._n, 1),
            "peak_speed": self._peak,
            "mean": self.mean.tolist(),
            "sigma": self.sigma.tolist(),
            "tau": self.tau,
        }

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        v = self._vec.to(pos.device).expand_as(pos)
        if self.seabed_z is None:
            return v.clone()
        return v * shear_profile(pos, self.seabed_z, self.layer)


def _across(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _axial(pos: torch.Tensor, origin: torch.Tensor, n: torch.Tensor):
    rel = pos - origin.to(pos.device)
    n = n.to(pos.device)
    x = (rel * n).sum(-1, keepdim=True)
    r = (rel - x * n).norm(dim=-1, keepdim=True)
    return x, r


class Jet:
    def __init__(self, point, direction, radius: float, outlet_velocity: float,
                 device: str = "cpu", *, decay: float = 5.8, spread: float = 0.094,
                 virtual_origin: float = 0.0):
        n = np.asarray(direction, dtype=np.float64)
        self.c = torch.as_tensor(np.asarray(point, dtype=np.float64),
                                 device=device, dtype=torch.float32)
        self.n = torch.as_tensor(n / np.linalg.norm(n), device=device,
                                 dtype=torch.float32)
        self.r = float(radius)
        self.vout = float(outlet_velocity)
        self.decay = float(decay)
        self.spread = float(spread)
        self.x0 = float(virtual_origin)

    def step(self, dt: float) -> None:
        pass

    def centreline(self, x):
        d = 2.0 * self.r
        return np.minimum(1.0, self.decay * d / np.maximum(np.asarray(x) - self.x0,
                                                           1e-9)) * self.vout

    def half_width(self, x):
        return np.maximum(self.r, self.spread * (np.asarray(x) - self.x0))

    def tracer_seed(self, k: int, rng) -> np.ndarray:
        c = self.c.detach().cpu().numpy()
        n = self.n.detach().cpu().numpy()
        e1, e2 = _across(n)
        x = rng.uniform(0.0, 25.0 * self.r, k)
        sigma = self.half_width(x) / np.sqrt(2.0 * np.log(2.0))
        rad = np.abs(rng.standard_normal(k)) * sigma
        th = rng.uniform(0.0, 2.0 * np.pi, k)
        return (c + x[:, None] * n + (rad * np.cos(th))[:, None] * e1
                + (rad * np.sin(th))[:, None] * e2)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        x, r = _axial(pos, self.c, self.n)
        d = 2.0 * self.r
        xs = (x - self.x0).clamp(min=1e-9)
        u0 = (self.decay * d / xs).clamp(max=1.0) * self.vout
        half = (self.spread * xs).clamp(min=self.r)
        u = u0 * torch.exp(-math.log(2.0) * (r / half) ** 2)
        return torch.where(x >= 0.0, u, torch.zeros_like(u)) * self.n.to(pos.device)


class Pipe:
    def __init__(self, p1, p2, r1: float, r2: float, v_in: float,
                 device: str = "cpu", *, profile: str = "auto", n: float = 7.0,
                 nu: float = 1.19e-6):
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        self.l = float(np.linalg.norm(b - a))
        if self.l < 1e-9:
            raise ValueError("pipe has zero length")
        if profile not in ("auto", "laminar", "turbulent"):
            raise ValueError(f"profile must be 'auto', 'laminar' or 'turbulent', "
                             f"not {profile!r}")
        self.p1 = torch.as_tensor(a, device=device, dtype=torch.float32)
        self.n = torch.as_tensor((b - a) / self.l, device=device, dtype=torch.float32)
        self.r1, self.r2 = float(r1), float(r2)
        self.vin = float(v_in)
        self.N = float(n)
        self.reynolds = abs(self.vin) * 2.0 * self.r1 / float(nu)
        if profile == "auto":
            profile = "laminar" if self.reynolds < 2300.0 else "turbulent"
        self.profile = profile

    def step(self, dt: float) -> None:
        pass

    def tracer_seed(self, k: int, rng) -> np.ndarray:
        p1 = self.p1.detach().cpu().numpy()
        n = self.n.detach().cpu().numpy()
        e1, e2 = _across(n)
        t = rng.uniform(0.0, self.l, k)
        r = self.r1 + (self.r2 - self.r1) * (t / self.l)
        rad = r * np.sqrt(rng.random(k)) * 0.9
        th = rng.uniform(0.0, 2.0 * np.pi, k)
        return (p1 + t[:, None] * n + (rad * np.cos(th))[:, None] * e1
                + (rad * np.sin(th))[:, None] * e2)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        x, rho = _axial(pos, self.p1, self.n)
        r = (self.r1 + (self.r2 - self.r1) * (x / self.l)).clamp(min=1e-9)
        inside = (x >= 0.0) & (x <= self.l) & (rho < r)
        bulk = self.vin * (self.r1 / r) ** 2
        eta = (rho / r).clamp(0.0, 1.0)
        if self.profile == "laminar":
            u = 2.0 * bulk * (1.0 - eta * eta)
        else:
            N = self.N
            u = bulk * (N + 1.0) * (2.0 * N + 1.0) / (2.0 * N * N) \
                * (1.0 - eta).pow(1.0 / N)
        return torch.where(inside, u, torch.zeros_like(u)) * self.n.to(pos.device)


class BoundaryLayer:
    def __init__(self, field, *, seabed_z: float, thickness: float = 1.5,
                 exponent: float = 1.0 / 7.0):
        self.field = field
        self.seabed_z = float(seabed_z)
        self.thickness = float(thickness)
        self.exponent = float(exponent)

    def step(self, dt: float) -> None:
        if hasattr(self.field, "step"):
            self.field.step(dt)

    @property
    def stats(self) -> dict:
        out = dict(getattr(self.field, "stats", {}) or {})
        out["boundary_layer"] = {"seabed_z": self.seabed_z,
                                 "thickness": self.thickness,
                                 "exponent": self.exponent}
        return out

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        return self.field(pos) * shear_profile(pos, self.seabed_z,
                                               self.thickness, self.exponent)


class Composite:
    def __init__(self, *fields):
        if not fields:
            raise ValueError("a Composite needs at least one field")
        self.fields = list(fields)

    def step(self, dt: float) -> None:
        for f in self.fields:
            if hasattr(f, "step"):
                f.step(dt)

    @property
    def stats(self) -> dict:
        out = {"fields": [type(f).__name__ for f in self.fields]}
        for f in self.fields:
            if hasattr(f, "stats"):
                out.update(f.stats)
        return out

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        v = self.fields[0](pos)
        for f in self.fields[1:]:
            v = v + f(pos)
        return v

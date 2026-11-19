"""Navigation sensor models: depth, DVL, altimeter, IMU and odometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PressureDepth", "DopplerVelocityLog", "Altimeter", "IMU", "Odometry"]


def _noise(shape, sigma, generator, device, dtype):
    if sigma <= 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype) * sigma


@dataclass
class PressureDepth:
    surface_height: float = 0.0
    density: float = 1025.0
    assumed_density: float | None = None
    sigma: float = 0.01
    resolution: float = 0.0
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __call__(self, pos_w: torch.Tensor, *, generator=None) -> torch.Tensor:
        up = torch.as_tensor(self.up, device=pos_w.device, dtype=pos_w.dtype)
        true_depth = (self.surface_height - (pos_w * up).sum(-1)).clamp(min=0.0)
        if self.assumed_density is not None:
            true_depth = true_depth * (self.density / float(self.assumed_density))
        out = true_depth + _noise(true_depth.shape, self.sigma, generator,
                                  pos_w.device, pos_w.dtype)
        if self.resolution > 0.0:
            out = torch.round(out / self.resolution) * self.resolution
        return out.clamp(min=0.0)

    def measure(self, state, *, generator=None) -> torch.Tensor:
        return self(state.pos_w, generator=generator)


@dataclass
class DopplerVelocityLog:
    max_altitude: float = 30.0
    sigma: float = 0.002
    scale_error: float = 0.002
    water_track: bool = False

    def __call__(self, lin_vel_b: torch.Tensor, altitude: torch.Tensor,
                 *, current_b: torch.Tensor | None = None, generator=None):
        v = lin_vel_b
        if self.water_track:
            if current_b is None:
                raise ValueError("water_track=True needs current_b, the current in "
                                 "the body frame")
            v = v - current_b

        valid = altitude <= self.max_altitude
        noise = _noise(v.shape, self.sigma, generator, v.device, v.dtype)
        out = v * (1.0 + self.scale_error) + noise
        return out * valid.unsqueeze(-1), valid

    def measure(self, state, *, generator=None):
        if state.altitude is None:
            raise ValueError(
                "a DVL needs altitude to know whether it has bottom lock; give "
                "the Vehicle a seabed_z")
        return self(state.lin_vel_b, state.altitude, current_b=state.current_b,
                    generator=generator)


@dataclass
class Altimeter:
    max_range: float = 50.0
    min_range: float = 0.3
    sigma: float = 0.01

    def __call__(self, altitude: torch.Tensor, *, generator=None):
        valid = (altitude <= self.max_range) & (altitude >= self.min_range)
        out = altitude + _noise(altitude.shape, self.sigma, generator,
                                altitude.device, altitude.dtype)
        return out.clamp(min=0.0) * valid, valid

    def measure(self, state, *, generator=None):
        if state.altitude is None:
            raise ValueError("an Altimeter needs altitude; give the Vehicle a seabed_z")
        return self(state.altitude, generator=generator)


@dataclass
class IMU:
    attitude_sigma: float = 0.0
    gyro_sigma: float = 0.0
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def measure(self, state, *, generator=None) -> torch.Tensor:
        from isaacswim.torch_math import quat_mul

        q = state.quat_w
        if self.attitude_sigma > 0.0:
            half = 0.5 * _noise(q.shape[:-1] + (3,), self.attitude_sigma, generator,
                                q.device, q.dtype)
            dq = torch.cat([torch.ones_like(half[..., :1]), half], dim=-1)
            q = quat_mul(q, dq / dq.norm(dim=-1, keepdim=True))
        w = state.ang_vel_b
        w = (w + torch.as_tensor(self.gyro_bias, device=w.device, dtype=w.dtype)
             + _noise(w.shape, self.gyro_sigma, generator, w.device, w.dtype))
        return torch.cat([q, w], dim=-1)


@dataclass
class Odometry:
    pos_sigma: float = 0.0

    def measure(self, state, *, generator=None) -> torch.Tensor:
        p = state.pos_w + _noise(state.pos_w.shape, self.pos_sigma, generator,
                                 state.pos_w.device, state.pos_w.dtype)
        return torch.cat([p, state.quat_w, state.lin_vel_b, state.ang_vel_b], dim=-1)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from isaacswim.np_math import quat_apply, quat_mul, quat_to_rpy, rpy_to_quat

__all__ = ["NavConfig", "Instruments", "DeadReckoning", "yaw_rotate"]


@dataclass
class NavConfig:
    dvl_hz: float = 10.0
    dvl_sigma: float = 0.003
    dvl_scale_sigma: float = 0.005
    heading_bias_deg: float = 0.5
    attitude_sigma: float = 0.005
    gyro_sigma: float = 0.002
    depth_sigma: float = 0.01
    depth_tau: float = 0.5
    dropout_tau: float = 1.0

    def scaled(self, k: float) -> NavConfig:
        return NavConfig(dvl_hz=self.dvl_hz, dvl_sigma=k * self.dvl_sigma,
                         dvl_scale_sigma=k * self.dvl_scale_sigma,
                         heading_bias_deg=k * self.heading_bias_deg,
                         attitude_sigma=k * self.attitude_sigma,
                         gyro_sigma=k * self.gyro_sigma, depth_sigma=k * self.depth_sigma,
                         depth_tau=self.depth_tau, dropout_tau=self.dropout_tau)


def yaw_rotate(q, yaw: float) -> np.ndarray:
    return quat_mul(np.asarray(rpy_to_quat((0.0, 0.0, yaw))), np.asarray(q, dtype=np.float64))


def _small_rotation(rng, sigma: float) -> np.ndarray:
    half = 0.5 * rng.normal(0.0, sigma, 3)
    q = np.array([1.0, *half])
    return q / np.linalg.norm(q)


class Instruments:
    def __init__(self, cfg: NavConfig, rng: np.random.Generator, *, dvl=None,
                 barometer=None, surface_z: float = 0.0, max_altitude: float = 30.0):
        self.cfg, self.rng = cfg, rng
        self.os_dvl, self.os_baro = dvl, barometer
        self.surface_z, self.max_altitude = float(surface_z), float(max_altitude)
        self.heading_bias = math.radians(rng.normal(0.0, cfg.heading_bias_deg))
        self.dvl_scale = 1.0 + rng.normal(0.0, cfg.dvl_scale_sigma)

    def ahrs(self, quat, ang_vel_b):
        q = quat_mul(np.asarray(quat, dtype=np.float64),
                     _small_rotation(self.rng, self.cfg.attitude_sigma))
        w = np.asarray(ang_vel_b, dtype=np.float64) + self.rng.normal(0, self.cfg.gyro_sigma, 3)
        return yaw_rotate(q, self.heading_bias), w

    def depth_z(self, pos, state=None) -> float:
        if self.os_baro is not None:
            return self.surface_z - float(self.os_baro.measure(state)[0])
        return float(pos[2]) + self.rng.normal(0.0, self.cfg.depth_sigma)

    def dvl(self, lin_vel_b, altitude: float, state=None):
        if self.os_dvl is not None:
            v, ok = self.os_dvl.measure(state)
            v, ok = v[0].detach().cpu().numpy().astype(np.float64), bool(ok[0])
        else:
            ok = altitude <= self.max_altitude
            v = np.asarray(lin_vel_b, dtype=np.float64) \
                + self.rng.normal(0.0, self.cfg.dvl_sigma, 3)
        return (v * self.dvl_scale if ok else np.zeros(3)), ok


class DeadReckoning:
    def __init__(self, cfg: NavConfig):
        self.cfg = cfg
        self.reset(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def reset(self, pos, quat) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.quat = np.asarray(quat, dtype=np.float64).copy()
        self.vel_b = np.zeros(3)
        self.ang_vel_b = np.zeros(3)
        self.lock = True

    @property
    def yaw(self) -> float:
        return quat_to_rpy(self.quat)[2]

    def update(self, dt: float, *, quat, ang_vel_b, depth_z: float,
               dvl=None, dvl_valid: bool = True) -> None:
        self.quat = np.asarray(quat, dtype=np.float64)
        self.ang_vel_b = np.asarray(ang_vel_b, dtype=np.float64)
        if dvl is not None:
            self.lock = bool(dvl_valid)
            if self.lock:
                self.vel_b = np.asarray(dvl, dtype=np.float64).copy()
        if not self.lock:
            self.vel_b = self.vel_b * math.exp(-dt / self.cfg.dropout_tau)
        v_w = quat_apply(self.quat, self.vel_b)
        self.pos[:2] += v_w[:2] * dt
        a = dt / (self.cfg.depth_tau + dt)
        self.pos[2] = (1.0 - a) * (self.pos[2] + v_w[2] * dt) + a * float(depth_z)

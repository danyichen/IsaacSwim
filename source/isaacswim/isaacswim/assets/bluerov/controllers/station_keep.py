from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import quat_apply_inverse as quat_rotate_inv  # noqa: F401
from isaacswim.np_math import quat_to_rpy, wrap_pi  # noqa: F401

__all__ = ["StationKeeper", "BatchStationKeeper", "HoldPose", "BLUEROV2_DAMPING",
           "quat_to_rpy", "quat_rotate_inv"]

BLUEROV2_DAMPING = ((13.7, 0.0, 33.0, 0.0), (141.0, 217.0, 190.0, 1.5))


class StationKeeper:
    def __init__(self, *,
                 kp=(250.0, 250.0, 250.0),
                 kd=(120.0, 120.0, 120.0),
                 ki=(25.0, 25.0, 25.0),
                 yaw_kp: float = 20.0,
                 yaw_kd: float = 8.0,
                 attitude_kp: float = 30.0,
                 attitude_kd: float = 10.0,
                 i_clamp: float = 25.0,
                 v_max: float = 0.35,
                 damping=None):
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64)
        self.yaw_kp = float(yaw_kp)
        self.yaw_kd = float(yaw_kd)
        self.attitude_kp = float(attitude_kp)
        self.attitude_kd = float(attitude_kd)
        self.i_clamp = float(i_clamp)
        self.v_max = float(v_max)
        d = np.zeros((2, 4)) if damping is None else np.asarray(damping, dtype=np.float64)
        self.damping = d.reshape(2, 4)
        self.reset()

    def reset(self):
        self._i = np.zeros(3)

    def command(self, *, pos_w, quat_w, lin_vel_b, ang_vel_b,
                target_pos, target_yaw, dt, feed_forward_b=None,
                target_vel=None, target_yaw_rate=0.0):
        pos_w = np.asarray(pos_w, dtype=np.float64)
        lin_vel_b = np.asarray(lin_vel_b, dtype=np.float64)
        ang_vel_b = np.asarray(ang_vel_b, dtype=np.float64)

        err_w = np.asarray(target_pos, dtype=np.float64) - pos_w
        err_b = quat_rotate_inv(quat_w, err_w)

        err_cap = np.where(self.kp > 0,
                           self.v_max * self.kd / np.maximum(self.kp, 1e-9),
                           np.inf)
        err_p = np.clip(err_b, -err_cap, err_cap)

        self._i += err_b * dt
        i_force = np.clip(self.ki * self._i, -self.i_clamp, self.i_clamp)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._i = np.where(self.ki > 0, i_force / np.maximum(self.ki, 1e-9), self._i)

        v_ref = np.zeros(3)
        if target_vel is not None:
            v_ref = quat_rotate_inv(quat_w, np.asarray(target_vel, dtype=np.float64))
        r_ref = float(target_yaw_rate)
        lin, quad = self.damping
        ref = np.append(v_ref, r_ref)
        drag = lin * ref + quad * np.abs(ref) * ref

        force_b = self.kp * err_p - self.kd * (lin_vel_b - v_ref) + i_force + drag[:3]
        if feed_forward_b is not None:
            force_b = force_b + np.asarray(feed_forward_b, dtype=np.float64)

        roll, pitch, yaw = quat_to_rpy(quat_w)
        yaw_err = wrap_pi(float(target_yaw) - yaw)

        moment_b = np.array([
            -self.attitude_kp * roll - self.attitude_kd * ang_vel_b[0],
            -self.attitude_kp * pitch - self.attitude_kd * ang_vel_b[1],
            self.yaw_kp * yaw_err - self.yaw_kd * (ang_vel_b[2] - r_ref) + drag[3],
        ])

        wrench = np.concatenate([force_b, moment_b])
        info = {
            "pos_err": err_b,
            "pos_err_norm": float(np.linalg.norm(err_w)),
            "yaw_err": yaw_err,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "i_force": i_force,
        }
        return wrench, info


class BatchStationKeeper:
    def __init__(self, num_envs: int, *, device="cpu", dtype=None,
                 kp=(250.0, 250.0, 250.0), kd=(120.0, 120.0, 120.0),
                 ki=(25.0, 25.0, 25.0), yaw_kp: float = 20.0, yaw_kd: float = 8.0,
                 attitude_kp: float = 30.0, attitude_kd: float = 10.0,
                 i_clamp: float = 25.0, v_max: float = 0.35):
        import torch
        self._torch = torch
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64
        t = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)  # noqa: E731
        self.kp, self.kd, self.ki = t(kp), t(kd), t(ki)
        self.yaw_kp, self.yaw_kd = float(yaw_kp), float(yaw_kd)
        self.attitude_kp, self.attitude_kd = float(attitude_kp), float(attitude_kd)
        self.i_clamp, self.v_max = float(i_clamp), float(v_max)
        inf = torch.full_like(self.kp, float("inf"))
        self._err_cap = torch.where(self.kp > 0,
                                    self.v_max * self.kd / self.kp.clamp_min(1e-9), inf)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None or not hasattr(self, "_i"):
            self._i = self._torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        else:
            self._i[env_ids] = 0.0

    def command(self, *, pos_w, quat_w, lin_vel_b, ang_vel_b, target_pos, target_yaw, dt,
                feed_forward_b=None):
        torch = self._torch
        f = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)  # noqa: E731
        pos_w, q, v_b, w_b = f(pos_w), f(quat_w), f(lin_vel_b), f(ang_vel_b)
        err_w = f(target_pos) - pos_w
        wq, u = q[:, 0:1], -q[:, 1:4]
        err_b = err_w + 2.0 * torch.cross(u, torch.cross(u, err_w, dim=-1) + wq * err_w, dim=-1)

        err_p = torch.maximum(torch.minimum(err_b, self._err_cap), -self._err_cap)
        self._i = self._i + err_b * dt
        i_force = (self.ki * self._i).clamp(-self.i_clamp, self.i_clamp)
        self._i = torch.where(self.ki > 0, i_force / self.ki.clamp_min(1e-9), self._i)

        force_b = self.kp * err_p - self.kd * v_b + i_force
        if feed_forward_b is not None:
            force_b = force_b + f(feed_forward_b)

        w, x, y, z = q.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_err = torch.remainder(f(target_yaw) - yaw + math.pi, 2.0 * math.pi) - math.pi

        moment_b = torch.stack([
            -self.attitude_kp * roll - self.attitude_kd * w_b[:, 0],
            -self.attitude_kp * pitch - self.attitude_kd * w_b[:, 1],
            self.yaw_kp * yaw_err - self.yaw_kd * w_b[:, 2],
        ], dim=-1)
        wrench = torch.cat([force_b, moment_b], dim=-1)
        info = {"pos_err": err_b, "pos_err_norm": torch.linalg.vector_norm(err_w, dim=-1),
                "yaw_err": yaw_err, "roll": roll, "pitch": pitch, "yaw": yaw,
                "i_force": i_force}
        return wrench, info


class HoldPose:
    def __init__(self, keeper: StationKeeper | BatchStationKeeper, target_pos,
                 target_yaw: float = 0.0, *, odom: str = "odom",
                 actuator: str = "thrusters"):
        self.keeper = keeper
        self.batched = isinstance(keeper, BatchStationKeeper)
        self.target_pos = target_pos if self.batched else np.asarray(target_pos, dtype=np.float64)
        self.target_yaw = target_yaw if self.batched else float(target_yaw)
        self.odom, self.actuator = odom, actuator
        self.info: dict = {}

    def reset(self, env_ids=None):
        if self.batched:
            self.keeper.reset(env_ids)
        else:
            self.keeper.reset()

    def __call__(self, obs, dt):
        if self.batched:
            o = obs[self.odom]
            wrench, self.info = self.keeper.command(
                pos_w=o[:, 0:3], quat_w=o[:, 3:7], lin_vel_b=o[:, 7:10], ang_vel_b=o[:, 10:13],
                target_pos=self.target_pos, target_yaw=self.target_yaw, dt=dt)
            return {self.actuator: wrench}
        o = obs[self.odom][0].detach().cpu().numpy().astype(np.float64)
        wrench, self.info = self.keeper.command(
            pos_w=o[0:3], quat_w=o[3:7], lin_vel_b=o[7:10], ang_vel_b=o[10:13],
            target_pos=self.target_pos, target_yaw=self.target_yaw, dt=dt)
        return {self.actuator: wrench}

from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import quat_apply, quat_apply_inverse, quat_to_mat

__all__ = [
    "ARM_JOINTS", "GRIP_JOINT", "ARM_STOW", "GRIP_OPEN", "GRIP_CLOSED", "JOINT_RATE",
    "TOOL_OFFSET", "TOOL_TIP_LINK", "TOOL_MOUNT_JOINT", "TOOL_GRASP", "FINGER_REACH",
    "GRASP_DEPTH", "CAMERA_TILT_WORK", "grip_angle_for_gap", "gripper_open",
    "tool_world", "vehicle_for_tool", "roll_to_close_across", "arm_pose_for",
    "jaw_direction_yaw", "SimpleArm",
]

ARM_JOINTS = ("alpha_axis_e", "alpha_axis_d", "alpha_axis_c", "alpha_axis_b")
GRIP_JOINT = "finger_joint"

ARM_STOW = {"alpha_axis_e": 2.85, "alpha_axis_d": 2.45, "alpha_axis_c": 1.80,
            "alpha_axis_b": 2.85}

GRIP_OPEN = 0.0
GRIP_CLOSED = 0.82
JOINT_RATE = 0.5

TOOL_OFFSET = 0.140
TOOL_TIP_LINK = "robotiq_tool_link"
TOOL_MOUNT_JOINT = "alpha_jaws_joint"
TOOL_GRASP = np.array([0.33, 0.0, -0.5525])
FINGER_REACH = 0.017
GRASP_DEPTH = 0.020
CAMERA_TILT_WORK = -0.55

_GRIP_GAP_MM = {0.000: 87.1, 0.103: 78.0, 0.205: 68.2, 0.308: 57.8, 0.410: 46.9,
                0.513: 35.6, 0.615: 24.0, 0.718: 12.4, 0.820: 0.7}


def grip_angle_for_gap(gap_m: float) -> float:
    qs = sorted(_GRIP_GAP_MM)
    gaps = [_GRIP_GAP_MM[q] for q in qs]
    return float(np.interp(-gap_m * 1e3, [-g for g in gaps], qs))


def gripper_open(q: float) -> bool:
    return q < GRIP_OPEN + 0.05


def tool_world(pos, quat, arm, q: dict) -> tuple[np.ndarray, np.ndarray]:
    p_b, R_b = arm.fk(q)
    R_v = quat_to_mat(quat)
    return np.asarray(pos, dtype=np.float64) + quat_apply(quat, p_b), R_v @ R_b


def vehicle_for_tool(tool_target_w, tool_base, quat) -> np.ndarray:
    return np.asarray(tool_target_w, dtype=np.float64) - quat_apply(quat, tool_base)


def roll_to_close_across(arm, q: dict, quat, direction_w) -> float | None:
    _, R0 = arm.fk({**q, "alpha_axis_b": 0.0})
    d = quat_apply_inverse(quat, np.asarray(direction_w, dtype=np.float64))
    d = d / (np.linalg.norm(d) + 1e-12)
    b = math.atan2(float(d @ R0[:, 0]), float(d @ R0[:, 1]))
    lo, hi = arm.limits["alpha_axis_b"]
    for cand in (b, b + math.pi, b - math.pi, b + 2 * math.pi, b - 2 * math.pi):
        if lo <= cand <= hi:
            return cand
    return None


def arm_pose_for(arm, tool_base, *, seed: dict, quat=(1.0, 0.0, 0.0, 0.0),
                 close_across_w=None) -> dict | None:
    q, _ = arm.ik(np.asarray(tool_base, dtype=np.float64), roll=0.0, seed=dict(seed))
    if q is None:
        return None
    if close_across_w is not None:
        b = roll_to_close_across(arm, q, quat, close_across_w)
        if b is not None:
            q["alpha_axis_b"] = b
    return q


def jaw_direction_yaw(arm, q: dict, quat) -> float:
    _, R = arm.fk(q)
    d = quat_apply(quat, R[:, 1])
    return math.atan2(float(d[1]), float(d[0]))


class SimpleArm:
    joints = ARM_JOINTS

    def __init__(self, *, mount=(0.10, 0.0, -0.10), l1: float = 0.25, l2: float = 0.30):
        self.mount = np.asarray(mount, dtype=np.float64)
        self.l1, self.l2 = float(l1), float(l2)
        self.limits = {"alpha_axis_e": (0.0, 5.7), "alpha_axis_d": (0.05, 3.1),
                       "alpha_axis_c": (0.05, 6.2), "alpha_axis_b": (0.0, 5.7)}

    def fk(self, q: dict) -> tuple[np.ndarray, np.ndarray]:
        yaw = q["alpha_axis_e"] - math.pi
        a1 = q["alpha_axis_d"] - math.pi
        a2 = q["alpha_axis_c"] - math.pi
        r = self.l1 * math.cos(a1) + self.l2 * math.cos(a1 + a2)
        z = self.l1 * math.sin(a1) + self.l2 * math.sin(a1 + a2)
        p = self.mount + np.array([r * math.cos(yaw), r * math.sin(yaw), z])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R0 = Rz @ np.diag([1.0, -1.0, -1.0])
        b = q["alpha_axis_b"]
        cb, sb = math.cos(-b), math.sin(-b)
        Rb = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        return p, R0 @ Rb

    def ik(self, target_base, *, roll=None, seed=None):
        seed = dict(seed or {})
        p = np.asarray(target_base, dtype=np.float64) - self.mount
        r = math.hypot(p[0], p[1])
        yaw = math.atan2(p[1], p[0])
        best, best_cost = None, float("inf")
        for sgn_r, yaw_b in ((1.0, yaw), (-1.0, yaw + math.pi)):
            e = self._wrap(yaw_b + math.pi, *self.limits["alpha_axis_e"])
            if e is None:
                continue
            x, z = sgn_r * r, p[2]
            dist = math.hypot(x, z)
            if dist > self.l1 + self.l2 or dist < abs(self.l1 - self.l2):
                continue
            c2 = (dist ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
            for elbow in (1.0, -1.0):
                a2 = elbow * math.acos(max(-1.0, min(1.0, c2)))
                a1 = math.atan2(z, x) - math.atan2(self.l2 * math.sin(a2),
                                                  self.l1 + self.l2 * math.cos(a2))
                d = self._wrap(a1 + math.pi, *self.limits["alpha_axis_d"])
                c = self._wrap(a2 + math.pi, *self.limits["alpha_axis_c"])
                if d is None or c is None:
                    continue
                b = seed.get("alpha_axis_b", 0.0) if roll is None else roll
                b = self._wrap(b, *self.limits["alpha_axis_b"])
                if b is None:
                    continue
                q = {"alpha_axis_e": e, "alpha_axis_d": d, "alpha_axis_c": c,
                     "alpha_axis_b": b}
                cost = sum((q[k] - seed[k]) ** 2 for k in q if k in seed)
                if cost < best_cost:
                    best, best_cost = q, cost
        if best is None:
            return None, {"reason": "unreachable"}
        return best, {"reason": "ok",
                      "error": float(np.linalg.norm(self.fk(best)[0] - (p + self.mount)))}

    @staticmethod
    def _wrap(q, lo, hi):
        for k in (0, 1, -1, 2, -2):
            v = q + 2 * math.pi * k
            if lo - 1e-9 <= v <= hi + 1e-9:
                return min(max(v, lo), hi)
        return None

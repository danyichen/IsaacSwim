from __future__ import annotations

import math

import numpy as np

__all__ = [
    "quat_apply",
    "quat_apply_inverse",
    "quat_mul",
    "quat_conj",
    "quat_to_mat",
    "quat_to_rpy",
    "rpy_to_quat",
    "rpy_to_mat",
    "quat_from_axis_angle",
    "quat_from_to",
    "wrap_pi",
]


def quat_apply(q, v) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, u = q[0], q[1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_apply_inverse(q, v) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, u = q[0], -q[1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = (float(x) for x in a)
    bw, bx, by, bz = (float(x) for x in b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_quat(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ])


def rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), *(axis * s)])


def quat_from_to(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a, b = a / na, b / nb
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1.0 + 1e-12:
        perp = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
        perp /= np.linalg.norm(perp)
        return np.array([0.0, *perp])
    axis = np.cross(a, b)
    q = np.array([1.0 + d, *axis])
    return q / np.linalg.norm(q)


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

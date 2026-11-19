from __future__ import annotations

import torch

__all__ = ["quat_apply", "quat_apply_inverse", "quat_mul", "quat_to_mat", "mat_to_quat"]


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = torch.cross(xyz, v, dim=-1)
    return v + 2.0 * (w * t + torch.cross(xyz, t, dim=-1))


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., 0:1]
    xyz = -q[..., 1:4]
    t = torch.cross(xyz, v, dim=-1)
    return v + 2.0 * (w * t + torch.cross(xyz, t, dim=-1))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_to_mat(q) -> torch.Tensor:
    q = torch.as_tensor(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def mat_to_quat(R) -> torch.Tensor:
    R = torch.as_tensor(R)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    sq = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                      1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1).clamp_min(0.0)
    cand = torch.stack([
        torch.stack([sq[..., 0], R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1),
        torch.stack([R[..., 2, 1] - R[..., 1, 2], sq[..., 1], R[..., 1, 0] + R[..., 0, 1],
                     R[..., 0, 2] + R[..., 2, 0]], -1),
        torch.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] + R[..., 0, 1], sq[..., 2],
                     R[..., 2, 1] + R[..., 1, 2]], -1),
        torch.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0],
                     R[..., 2, 1] + R[..., 1, 2], sq[..., 3]], -1),
    ], -2)
    k = sq.argmax(-1)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)

"""Acceleration added mass, applied stably on top of the physics step."""

from __future__ import annotations

import torch

from isaacswim import wrench_warp
from isaacswim.torch_math import quat_to_mat

wp = wrench_warp.wp

__all__ = ["AddedMass"]


class AddedMass:
    def __init__(self, added, mass, inertia, dt: float, num_envs: int,
                 device="cpu", dtype=torch.float32, teleport_tol: float = 0.01,
                 body_com=None, backend: str = "auto"):
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.device, self.dtype = torch.device(device), dtype
        A = torch.as_tensor(added, dtype=torch.float64).cpu()
        m = torch.as_tensor(mass, dtype=torch.float64).cpu()
        I = torch.as_tensor(inertia, dtype=torch.float64).cpu()
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if I.dim() == 3:
            I = I.unsqueeze(0)
        B = A.shape[0]
        M = torch.zeros(*m.shape, 6, 6, dtype=torch.float64)
        M[..., 0, 0] = M[..., 1, 1] = M[..., 2, 2] = m
        M[..., 3:, 3:] = I
        P = torch.linalg.solve((M + A).transpose(-1, -2), A.expand_as(M).transpose(-1, -2))
        self.P = P.transpose(-1, -2).to(self.device, dtype)
        self.mass = m.to(self.device, dtype)
        self.inertia = I.to(self.device, dtype)
        self.num_bodies = B
        E = self.num_envs
        self._vel = torch.zeros(E, B, 6, device=self.device, dtype=dtype)
        self._applied_w = torch.zeros_like(self._vel)
        self._valid = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._pos = None
        self.teleport_tol = float(teleport_tol)
        self.body_com = None
        if body_com is not None:
            c = torch.as_tensor(body_com, dtype=dtype, device=self.device)
            self.body_com = (c.unsqueeze(0) if c.dim() == 2 else c).contiguous()

        if backend not in ("auto", "torch", "warp"):
            raise ValueError(f"backend must be 'auto', 'torch' or 'warp', got {backend!r}")
        self.backend = "torch"
        self._warp = None
        if backend != "torch" and dtype == torch.float32 and wrench_warp.available():
            self._warp = _WarpAddedMass(self)
            self.backend = "warp"
        elif backend == "warp":
            raise ValueError("backend='warp' needs NVIDIA Warp and dtype=torch.float32")

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._valid[:] = False
        else:
            self._valid[env_ids] = False

    def force(self, quat_w: torch.Tensor, com_vel_w: torch.Tensor,
              F: torch.Tensor, T: torch.Tensor, pos_w: torch.Tensor | None = None):
        if self._warp is not None:
            return self._warp.force(quat_w, com_vel_w, F, T, pos_w)
        R = quat_to_mat(quat_w)
        Rt = R.transpose(-1, -2)
        K = torch.cat([F, T], -1)
        if pos_w is not None:
            com_pos_w = pos_w
            if self.body_com is not None:
                com_pos_w = pos_w + (R @ self.body_com.unsqueeze(-1)).squeeze(-1)
            if self._pos is not None:
                jump = com_pos_w - self._pos - com_vel_w[..., :3] * self.dt
                self._valid &= jump.norm(dim=-1).amax(-1) < self.teleport_tol
            self._pos = com_pos_w.detach().clone()

        a = (com_vel_w - self._vel) / self.dt
        alpha_l = (Rt @ a[..., 3:].unsqueeze(-1))
        Ia = (R @ (self.inertia @ alpha_l)).squeeze(-1)
        Ma = torch.cat([self.mass.unsqueeze(-1) * a[..., :3], Ia], -1)
        G_w = (Ma - self._applied_w) * self._valid.view(-1, 1, 1)
        G_l = torch.cat([(Rt @ G_w[..., :3].unsqueeze(-1)).squeeze(-1),
                         (Rt @ G_w[..., 3:].unsqueeze(-1)).squeeze(-1)], -1)

        f = -(self.P @ (K + G_l).unsqueeze(-1)).squeeze(-1)
        total = K + f
        self._applied_w = torch.cat([(R @ total[..., :3].unsqueeze(-1)).squeeze(-1),
                                     (R @ total[..., 3:].unsqueeze(-1)).squeeze(-1)], -1)
        self._vel = com_vel_w.detach().clone()
        self._valid[:] = True
        return f[..., :3], f[..., 3:]


if wp is not None:
    @wp.kernel
    def _added_mass_kernel(
        quat: wp.array2d(dtype=wp.vec4), vel: wp.array2d(dtype=wp.spatial_vector),
        F: wp.array2d(dtype=wp.vec3), T: wp.array2d(dtype=wp.vec3),
        pos: wp.array2d(dtype=wp.vec3), use_pos: int,
        com_b: wp.array2d(dtype=wp.vec3), com_per_env: int,
        P: wp.array2d(dtype=wp.spatial_matrix), mass: wp.array2d(dtype=float),
        inertia: wp.array2d(dtype=wp.mat33), per_env: int, num_bodies: int,
        dt: float, tol: float,
        vel_prev: wp.array2d(dtype=wp.spatial_vector),
        applied: wp.array2d(dtype=wp.spatial_vector),
        pos_prev: wp.array2d(dtype=wp.vec3), valid: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=wp.spatial_vector),
    ):
        e = wp.tid()
        pe = int(0)  # noqa: UP018 -- Warp: a typed local
        if per_env != 0:
            pe = e
        ce = int(0)  # noqa: UP018
        if com_per_env != 0:
            ce = e
        ok = float(valid[e])
        if use_pos != 0:
            for b in range(num_bodies):
                q = quat[e, b]
                com = pos[e, b] + wp.quat_rotate(wp.quat(q[1], q[2], q[3], q[0]), com_b[ce, b])
                if use_pos == 2:
                    jump = com - pos_prev[e, b] - wp.spatial_top(vel[e, b]) * dt
                    if wp.length(jump) >= tol:
                        ok = 0.0
                pos_prev[e, b] = com
        for b in range(num_bodies):
            q = quat[e, b]
            R = wp.quat_to_matrix(wp.quat(q[1], q[2], q[3], q[0]))
            Rt = wp.transpose(R)
            v = vel[e, b]
            a = (v - vel_prev[e, b]) / dt
            Ia = R * (inertia[pe, b] * (Rt * wp.spatial_bottom(a)))
            Ma = wp.spatial_vector(mass[pe, b] * wp.spatial_top(a), Ia)
            G_w = (Ma - applied[e, b]) * ok
            G_l = wp.spatial_vector(Rt * wp.spatial_top(G_w), Rt * wp.spatial_bottom(G_w))
            K = wp.spatial_vector(F[e, b], T[e, b])
            f = -(P[pe, b] * (K + G_l))
            total = K + f
            applied[e, b] = wp.spatial_vector(R * wp.spatial_top(total),
                                              R * wp.spatial_bottom(total))
            vel_prev[e, b] = v
            out[e, b] = f
        valid[e] = 1


class _WarpAddedMass:
    def __init__(self, am: AddedMass):
        self.device = wp.device_from_torch(am.device)
        E, B, dev = am.num_envs, am.num_bodies, am.device

        def z(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        self.t = t = {"quat": z(E, B, 4), "vel": z(E, B, 6), "F": z(E, B, 3), "T": z(E, B, 3),
                      "pos": z(E, B, 3), "pos_prev": z(E, B, 3), "out": z(E, B, 6),
                      "vel_prev": z(E, B, 6), "applied": z(E, B, 6),
                      "valid": z(E, dtype=torch.int32)}
        self.P = am.P.contiguous()
        self.com_b = am.body_com if am.body_com is not None else z(1, B, 3)
        self.mass = am.mass.contiguous()
        self.inertia = am.inertia.contiguous()
        vec = {3: wp.vec3, 4: wp.vec4, 6: wp.spatial_vector}
        w = {k: wp.from_torch(v, dtype=vec[v.shape[-1]] if v.dim() == 3 else wp.int32)
             for k, v in t.items()}
        self.has_prev = False
        self.cmd = wp.launch(_added_mass_kernel, dim=E, device=self.device, record_cmd=True,
                             inputs=[w["quat"], w["vel"], w["F"], w["T"], w["pos"], 0,
                                     wp.from_torch(self.com_b, dtype=wp.vec3),
                                     int(self.com_b.shape[0] > 1),
                                     wp.from_torch(self.P, dtype=wp.spatial_matrix),
                                     wp.from_torch(self.mass, dtype=wp.float32),
                                     wp.from_torch(self.inertia, dtype=wp.mat33),
                                     int(self.P.shape[0] > 1), B, am.dt, am.teleport_tol,
                                     w["vel_prev"], w["applied"], w["pos_prev"], w["valid"],
                                     w["out"]])
        self.use_pos = 0
        am._valid = t["valid"]

    def force(self, quat_w, com_vel_w, F, T, pos_w):
        t = self.t
        t["quat"].copy_(quat_w)
        t["vel"].copy_(com_vel_w)
        t["F"].copy_(F)
        t["T"].copy_(T)
        use_pos = 0
        if pos_w is not None:
            t["pos"].copy_(pos_w)
            use_pos = 2 if self.has_prev else 1
            self.has_prev = True
        if use_pos != self.use_pos:
            self.cmd.set_param_by_name("use_pos", use_pos)
            self.use_pos = use_pos
        stream = None
        if self.device.is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(quat_w.device))
        self.cmd.launch(stream)
        out = t["out"].clone()
        return out[..., :3], out[..., 3:]

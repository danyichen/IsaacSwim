"""The hydrodynamic force and torque on every shape, batched over environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from isaacswim.medium import Medium
from isaacswim.meshvolume import MeshDisplacement
from isaacswim.shapes import FluidShape
from isaacswim.torch_math import quat_to_mat

__all__ = ["HydroWrench"]

MJ_MINVAL = 1e-15


class HydroWrench:
    def __init__(self, shapes: Sequence[FluidShape], body_names: Sequence[str],
                 medium: Medium | None = None, *, body_com=None,
                 buoyancy: bool = True, mesh_buoyancy: bool = True,
                 drag_scales_with_submersion: bool = True,
                 minval: float = MJ_MINVAL,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                 backend: str = "auto"):
        self.medium = medium if medium is not None else Medium()
        self.device = torch.device(device)
        self.dtype = dtype
        self.body_names = list(body_names)
        self.num_bodies = len(self.body_names)
        self.minval = float(minval)
        self.shapes = [s for s in shapes if s.enabled]
        if not self.shapes:
            raise ValueError("no enabled shapes were given")

        idx_of = {n: i for i, n in enumerate(self.body_names)}
        missing = sorted({s.body for s in self.shapes} - set(idx_of))
        if missing:
            raise KeyError(
                f"shapes reference bodies that the simulator does not have: {missing}. "
                f"Known bodies: {self.body_names}")

        t = lambda a, d=dtype: torch.as_tensor(np.asarray(a), dtype=d, device=self.device)  # noqa: E731
        self.bidx = t([idx_of[s.body] for s in self.shapes], torch.long)
        self.n_slots = len(self.shapes)

        semi = np.stack([s.semiaxes for s in self.shapes])
        self.semi = t(semi)
        self.slot_pos = t(np.stack([s.pos for s in self.shapes]))
        self.vmass = t(np.stack([s.virtual_mass for s in self.shapes]))
        self.vinert = t(np.stack([s.virtual_inertia for s in self.shapes]))
        self.volume_disp = t(np.array([s.volume for s in self.shapes]))

        coef = t(np.stack([s.coef for s in self.shapes]))
        self.c_blunt, self.c_slender = coef[:, 0], coef[:, 1]
        self.c_ang, self.c_kutta, self.c_magnus = coef[:, 2], coef[:, 3], coef[:, 4]

        quats = np.stack([s.quat for s in self.shapes])
        R_ls = _quat_to_mat_np(quats)
        self.R_ls = t(R_ls)
        self.shapes_are_axis_aligned = bool(np.allclose(R_ls, np.eye(3), atol=1e-12))

        self.set_body_com(body_com)

        s64 = semi.astype(np.float64)
        d_max = s64.max(axis=1)
        d_min = s64.min(axis=1)
        d_mid = s64.sum(axis=1) - d_max - d_min
        self.A_max = t(np.pi * d_max * d_mid)
        self.volume = t(4.0 / 3.0 * np.pi * s64.prod(axis=1))
        self.I_max = t(8.0 / 15.0 * np.pi * d_mid * d_max ** 4)
        self.II = t(np.stack([
            8.0 / 15.0 * np.pi * s64[:, k]
            * np.maximum(s64[:, (k + 1) % 3], s64[:, (k + 2) % 3]) ** 4
            for k in range(3)
        ], axis=1))
        eq_D = 2.0 / 3.0 * s64.sum(axis=1)
        self.lin_visc_force = t(3.0 * np.pi * eq_D)
        self.lin_visc_torq = t(np.pi * eq_D ** 3)
        inv_dmax = np.where(d_max > minval, 1.0 / np.maximum(d_max, minval), 0.0)
        sn = s64 * inv_dmax[:, None]
        self.d_max = t(d_max)
        self.ab2_n = t(np.stack([sn[:, 1] * sn[:, 2],
                                 sn[:, 2] * sn[:, 0],
                                 sn[:, 0] * sn[:, 1]], axis=1) ** 2)

        g = np.asarray(self.medium.gravity, dtype=np.float64)
        self.gravity = t(g)
        self.buoyancy = bool(buoyancy) and float(np.linalg.norm(g)) > 0.0 \
            and float(self.volume_disp.sum()) > 0.0
        self.drag_scales_with_submersion = bool(drag_scales_with_submersion)

        meshes = [s.mesh for s in self.shapes]
        self.mesh_disp = None
        self.mesh_cob = None
        if mesh_buoyancy and self.buoyancy and any(m is not None for m in meshes):
            self.mesh_disp = MeshDisplacement(meshes, device=self.device, dtype=dtype)
            self.mesh_cob = torch.where(self.mesh_disp.has_mesh.unsqueeze(-1),
                                        self.mesh_disp.centroid,
                                        torch.zeros_like(self.mesh_disp.centroid)
                                        ).unsqueeze(0)
        self.free_surface = self.medium.surface_height is not None
        self._surface_z = float(self.medium.surface_height or 0.0)
        self._const_current = t(np.asarray(self.medium.current, dtype=np.float64)).view(1, 1, 3)
        self._has_current = bool(np.any(np.asarray(self.medium.current) != 0.0))
        g_norm = float(np.linalg.norm(g))
        self._up = t(-g / g_norm) if g_norm > 0.0 else t([0.0, 0.0, 1.0])

        deep = s64.max(axis=1).copy()
        for i, m in enumerate(meshes):
            if m is not None and self.mesh_disp is not None:
                deep[i] = max(deep[i], float(m.radius))
        self._deep_depth = t(deep).view(1, -1)

        if backend not in ("auto", "torch", "warp"):
            raise ValueError(f"backend must be 'auto', 'torch' or 'warp', got {backend!r}")
        self.backend = "torch"
        self._warp = None
        if backend != "torch" and dtype == torch.float32:
            from isaacswim import wrench_warp
            if wrench_warp.available():
                self._warp = wrench_warp.WarpWrench(self)
                self.backend = "warp"
            elif backend == "warp":
                raise ImportError("backend='warp' needs NVIDIA Warp (`pip install warp-lang`; "
                                  "Isaac Sim ships it)")
        elif backend == "warp":
            raise ValueError("backend='warp' is float32 only; use dtype=torch.float32")

    def set_body_com(self, body_com) -> None:
        if body_com is None:
            self.arm = self.slot_pos.unsqueeze(0)
            return
        com = torch.as_tensor(body_com, dtype=self.dtype, device=self.device)
        if com.dim() == 2:
            com = com.unsqueeze(0)
        if com.shape[-2] != self.num_bodies:
            raise ValueError(f"body_com must be (...,{self.num_bodies},3), got {tuple(com.shape)}")
        self.arm = self.slot_pos.unsqueeze(0) - com[:, self.bidx]

    @property
    def density(self) -> float:
        return self.medium.density

    def set_medium(self, medium: Medium) -> None:
        self.medium = medium
        self.gravity = torch.as_tensor(np.asarray(medium.gravity), dtype=self.dtype,
                                       device=self.device)
        self.free_surface = medium.surface_height is not None
        self._surface_z = float(medium.surface_height or 0.0)
        self._const_current = torch.as_tensor(np.asarray(medium.current), dtype=self.dtype,
                                              device=self.device).view(1, 1, 3)
        self._has_current = bool(np.any(np.asarray(medium.current) != 0.0))

    def slot_wrench(self, v: torch.Tensor, w: torch.Tensor):
        rho, mu = self.medium.density, self.medium.viscosity

        p_lin = rho * self.vmass * v
        p_ang = rho * self.vinert * w
        f = torch.cross(p_lin, w, dim=-1)
        tau = torch.cross(p_lin, v, dim=-1) + torch.cross(p_ang, w, dim=-1)

        f = f + (self.c_magnus * rho * self.volume).unsqueeze(-1) * torch.cross(w, v, dim=-1)

        v_norm = torch.linalg.vector_norm(v, dim=-1)
        inv_speed = torch.where(v_norm > self.minval, 1.0 / v_norm.clamp_min(self.minval),
                                torch.zeros_like(v_norm))
        vh = v * inv_speed.unsqueeze(-1)
        vh2 = vh ** 2
        proj_num = (self.ab2_n * vh2).sum(-1)
        inv_num = torch.where(proj_num > self.minval, 1.0 / proj_num.clamp_min(self.minval),
                              torch.zeros_like(proj_num))
        abc = self.ab2_n * inv_num.unsqueeze(-1)
        proj_den = (abc ** 2 * vh2).sum(-1)
        A_proj = np.pi * self.d_max ** 2 * torch.sqrt(proj_num * proj_den)

        normv = abc * vh
        cos_alpha = torch.where(proj_den > self.minval, 1.0 / proj_den.clamp_min(self.minval),
                                torch.zeros_like(proj_den))
        circ = (self.c_kutta * rho * cos_alpha * A_proj).unsqueeze(-1) \
            * torch.cross(normv, v, dim=-1)
        f = f + torch.cross(circ, v, dim=-1)

        mom_visc = w * (self.c_ang.unsqueeze(-1) * self.II
                        + self.c_slender.unsqueeze(-1) * (self.I_max.unsqueeze(-1) - self.II))
        drag_lin = (mu * self.lin_visc_force
                    + rho * v_norm * (A_proj * self.c_blunt
                                      + self.c_slender * (self.A_max - A_proj)))
        drag_ang = (mu * self.lin_visc_torq
                    + rho * torch.linalg.vector_norm(mom_visc, dim=-1))

        f = f - drag_lin.unsqueeze(-1) * v
        tau = tau - drag_ang.unsqueeze(-1) * w
        return f, tau

    def submerged_fraction(self, depth: torch.Tensor, up_shape: torch.Tensor):
        m = self.semi * up_shape
        h = torch.linalg.vector_norm(m, dim=-1).clamp_min(MJ_MINVAL)
        t = (depth / h).clamp(-1.0, 1.0)

        frac = (2.0 + 3.0 * t - t ** 3) * 0.25
        vol_ball = t - t ** 3 / 3.0 + 2.0 / 3.0
        z_bar = (t * t * 0.5 - t ** 4 * 0.25 - 0.25) / vol_ball.clamp_min(1e-9)
        z_bar = torch.where(t >= 1.0 - 1e-6, torch.zeros_like(z_bar), z_bar)

        y_bar = (z_bar / h).unsqueeze(-1) * m
        return frac, self.semi * y_bar

    def compute(self, quat_w: torch.Tensor, lin_vel_w: torch.Tensor,
                ang_vel_w: torch.Tensor, pos_w: torch.Tensor | None = None,
                current: torch.Tensor | Callable | None = None,
                surface: Callable | None = None, vel_at_com: bool = False):
        if self._warp is not None and quat_w.dtype == torch.float32:
            return self._warp.compute(self, quat_w, lin_vel_w, ang_vel_w, pos_w, current, surface,
                                      vel_at_com)

        q = quat_w[:, self.bidx]
        v_org = lin_vel_w[:, self.bidx]
        w_w = ang_vel_w[:, self.bidx]
        num_envs = v_org.shape[0]

        R_wl = quat_to_mat(q)
        R_ws = R_wl if self.shapes_are_axis_aligned else R_wl @ self.R_ls

        r_w = (R_wl @ self.slot_pos.unsqueeze(-1)).squeeze(-1)
        r_v = (R_wl @ self.arm.unsqueeze(-1)).squeeze(-1) if vel_at_com else r_w
        v_w = v_org + torch.cross(w_w, r_v, dim=-1)

        free_surface = self.free_surface or surface is not None
        need_pos = (self.buoyancy and free_surface) or callable(current)
        centre_w = None
        if pos_w is not None:
            centre_w = pos_w[:, self.bidx] + r_w
        elif need_pos:
            raise ValueError("pos_w is required for a free surface or a current field")

        if callable(current):
            v_w = v_w - current(centre_w)
        elif current is not None:
            v_w = v_w - torch.as_tensor(current, dtype=v_w.dtype, device=v_w.device)
        elif self._has_current:
            v_w = v_w - self._const_current

        cols = [v_w, w_w]
        if self.buoyancy:
            cols += [self._up.expand_as(v_w), self.gravity.expand_as(v_w)]
        X_s = R_ws.transpose(-1, -2) @ torch.stack(cols, dim=-1)
        v_s, w_s = X_s[..., 0], X_s[..., 1]

        f_s, tau_s = self.slot_wrench(v_s, w_s)

        arm = self.arm

        f_buoy_s = None
        if self.buoyancy:
            up_s, g_s = X_s[..., 2], X_s[..., 3]

            if free_surface:
                level = self._surface_z if surface is None else surface(centre_w)
                depth = level - (centre_w * self._up).sum(-1)
                check = self.mesh_disp is not None or depth.device.type == "cpu"
                if check and bool((depth >= self._deep_depth).all()):
                    cob_s = self.mesh_cob
                    vol = self.volume_disp.expand(num_envs, -1)
                else:
                    frac, cob_s = self.submerged_fraction(depth, up_s)
                    if self.mesh_disp is not None:
                        m_frac, m_cob = self.mesh_disp.submerged(depth, up_s)
                        has = self.mesh_disp.has_mesh.view(1, -1)
                        frac = torch.where(has, m_frac, frac)
                        cob_s = torch.where(has.unsqueeze(-1), m_cob, cob_s)
                    if self.drag_scales_with_submersion:
                        f_s = f_s * frac.unsqueeze(-1)
                        tau_s = tau_s * frac.unsqueeze(-1)
                    vol = self.volume_disp * frac
            else:
                cob_s = self.mesh_cob
                vol = self.volume_disp.expand(num_envs, -1)

            f_buoy_s = (-self.medium.density * vol).unsqueeze(-1) * g_s
            if cob_s is not None:
                tau_s = tau_s + torch.cross(cob_s.expand_as(f_buoy_s), f_buoy_s, dim=-1)
            f_s = f_s + f_buoy_s

        if self.shapes_are_axis_aligned:
            f_l, tau_l = f_s, tau_s
        else:
            FT_l = self.R_ls @ torch.stack([f_s, tau_s], dim=-1)
            f_l, tau_l = FT_l[..., 0], FT_l[..., 1]
        tau_l = tau_l + torch.cross(arm.expand_as(f_l), f_l, dim=-1)

        F = torch.zeros((num_envs, self.num_bodies, 3), device=f_l.device, dtype=f_l.dtype)
        T = torch.zeros_like(F)
        F.index_add_(1, self.bidx, f_l)
        T.index_add_(1, self.bidx, tau_l)
        return F, T

    def added_mass_matrix(self) -> torch.Tensor:
        rho = float(self.medium.density)
        R = self.R_ls.double().cpu()
        mt = R @ torch.diag_embed(rho * self.vmass.double().cpu()) @ R.transpose(-1, -2)
        mr = R @ torch.diag_embed(rho * self.vinert.double().cpu()) @ R.transpose(-1, -2)
        c = self.arm[0].double().cpu()
        X = torch.zeros(self.n_slots, 3, 3, dtype=torch.float64)
        X[:, 0, 1], X[:, 0, 2], X[:, 1, 2] = -c[:, 2], c[:, 1], -c[:, 0]
        X = X - X.transpose(-1, -2)
        A = torch.zeros(self.num_bodies, 6, 6, dtype=torch.float64)
        blocks = torch.cat([torch.cat([mt, -mt @ X], -1),
                            torch.cat([X @ mt, mr - X @ mt @ X], -1)], -2)
        A.index_add_(0, self.bidx.cpu(), blocks)
        return A

    def __repr__(self) -> str:
        meshed = 0 if self.mesh_disp is None else int(self.mesh_disp.has_mesh_np.sum())
        return (f"HydroWrench({self.n_slots} shapes on {self.num_bodies} bodies, "
                f"rho={self.medium.density:g}, buoyancy={self.buoyancy}"
                + (f" (mesh on {meshed}/{self.n_slots})" if meshed else "")
                + f", free_surface={self.free_surface}, device={self.device})")


def _quat_to_mat_np(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)

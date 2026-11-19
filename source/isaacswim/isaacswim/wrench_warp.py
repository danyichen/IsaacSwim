from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

try:  # pragma: no cover
    import warp as wp
    _WARP = True
except Exception:  # pragma: no cover
    wp = None
    _WARP = False

__all__ = ["available", "WarpWrench"]


def available() -> bool:
    if not _WARP:
        return False
    try:
        wp.init()
        return True
    except Exception:  # pragma: no cover
        return False


if _WARP:
    wp.config.quiet = True

    @wp.func
    def _qrot(q: wp.vec4, v: wp.vec3) -> wp.vec3:
        u = wp.vec3(q[1], q[2], q[3])
        t = wp.cross(u, v)
        return v + 2.0 * (q[0] * t + wp.cross(u, t))

    @wp.func
    def _qrot_inv(q: wp.vec4, v: wp.vec3) -> wp.vec3:
        u = wp.vec3(-q[1], -q[2], -q[3])
        t = wp.cross(u, v)
        return v + 2.0 * (q[0] * t + wp.cross(u, t))

    @wp.func
    def _safe_inv(x: float, minval: float) -> float:
        r = float(0.0)  # noqa: UP018 -- Warp: a typed local, reassigned below
        if x > minval:
            r = 1.0 / wp.max(x, minval)
        return r

    @wp.func
    def _cap(depth: float, semi: wp.vec3, up_s: wp.vec3):
        m = wp.cw_mul(semi, up_s)
        h = wp.max(wp.length(m), 1.0e-15)
        t = wp.clamp(depth / h, -1.0, 1.0)
        frac = (2.0 + 3.0 * t - t * t * t) * 0.25
        vol_ball = t - t * t * t / 3.0 + 2.0 / 3.0
        z_bar = (t * t * 0.5 - t * t * t * t * 0.25 - 0.25) / wp.max(vol_ball, 1.0e-9)
        if t >= 1.0 - 1.0e-6:
            z_bar = 0.0
        y_bar = (z_bar / h) * m
        return frac, wp.cw_mul(semi, y_bar)

    @wp.kernel
    def _depth_kernel(
        quat: wp.array2d(dtype=wp.vec4), pos: wp.array2d(dtype=wp.vec3),
        bidx: wp.array(dtype=wp.int32), slot_pos: wp.array(dtype=wp.vec3),
        R_ls: wp.array(dtype=wp.mat33), up: wp.vec3, surface_z: float,
        level: wp.array2d(dtype=float), use_level: int,
        radius: wp.array(dtype=float), has_mesh: wp.array(dtype=wp.int32),
        depth_out: wp.array2d(dtype=float), up_s_out: wp.array2d(dtype=wp.vec3),
        n_cut: wp.array(dtype=wp.int32),
    ):
        e, s = wp.tid()
        b = bidx[s]
        q = quat[e, b]
        centre = pos[e, b] + _qrot(q, slot_pos[s])
        lvl = surface_z
        if use_level != 0:
            lvl = level[e, s]
        d = lvl - wp.dot(centre, up)
        depth_out[e, s] = d
        up_s_out[e, s] = wp.transpose(R_ls[s]) * _qrot_inv(q, up)
        if has_mesh[s] != 0:
            if d > -radius[s] and d < radius[s]:
                wp.atomic_add(n_cut, 0, 1)

    @wp.kernel
    def _wrench_kernel(
        quat: wp.array2d(dtype=wp.vec4), lin_vel: wp.array2d(dtype=wp.vec3),
        ang_vel: wp.array2d(dtype=wp.vec3),
        current: wp.array2d(dtype=wp.vec3), current_const: wp.vec3, use_current: int,
        bidx: wp.array(dtype=wp.int32), slot_pos: wp.array(dtype=wp.vec3),
        R_ls: wp.array(dtype=wp.mat33), arm: wp.array2d(dtype=wp.vec3), arm_per_env: int,
        vel_at_com: int,
        semi: wp.array(dtype=wp.vec3), vmass: wp.array(dtype=wp.vec3),
        vinert: wp.array(dtype=wp.vec3), coef: wp.array2d(dtype=float),
        volume: wp.array(dtype=float), d_max: wp.array(dtype=float),
        ab2_n: wp.array(dtype=wp.vec3), A_max: wp.array(dtype=float),
        I_max: wp.array(dtype=float), II: wp.array(dtype=wp.vec3),
        lin_visc_force: wp.array(dtype=float), lin_visc_torq: wp.array(dtype=float),
        rho: float, mu: float, minval: float,
        buoyancy: int, free_surface: int, drag_scales: int,
        gravity: wp.vec3, up: wp.vec3,
        volume_disp: wp.array(dtype=float),
        depth: wp.array2d(dtype=float), up_s_in: wp.array2d(dtype=wp.vec3),
        has_mesh: wp.array(dtype=wp.int32), mesh_radius: wp.array(dtype=float),
        mesh_centroid: wp.array(dtype=wp.vec3),
        use_override: int, frac_override: wp.array2d(dtype=float),
        cob_override: wp.array2d(dtype=wp.vec3),
        F: wp.array2d(dtype=wp.vec3), T: wp.array2d(dtype=wp.vec3),
    ):
        e, s = wp.tid()
        b = bidx[s]
        q = quat[e, b]
        Rls = R_ls[s]
        RlsT = wp.transpose(Rls)

        ea = int(0)  # noqa: UP018
        if arm_per_env != 0:
            ea = e

        w_w = ang_vel[e, b]
        r_v = _qrot(q, slot_pos[s])
        if vel_at_com != 0:
            r_v = _qrot(q, arm[ea, s])
        v_w = lin_vel[e, b] + wp.cross(w_w, r_v)
        if use_current == 1:
            v_w = v_w - current_const
        elif use_current == 2:
            v_w = v_w - current[e, s]

        v = RlsT * _qrot_inv(q, v_w)
        w = RlsT * _qrot_inv(q, w_w)

        p_lin = rho * wp.cw_mul(vmass[s], v)
        p_ang = rho * wp.cw_mul(vinert[s], w)
        f = wp.cross(p_lin, w)
        tau = wp.cross(p_lin, v) + wp.cross(p_ang, w)

        c_blunt = coef[s, 0]
        c_slender = coef[s, 1]
        c_ang = coef[s, 2]
        c_kutta = coef[s, 3]
        c_magnus = coef[s, 4]
        f = f + (c_magnus * rho * volume[s]) * wp.cross(w, v)

        v_norm = wp.length(v)
        vh = v * _safe_inv(v_norm, minval)
        vh2 = wp.cw_mul(vh, vh)
        ab2 = ab2_n[s]
        proj_num = wp.dot(ab2, vh2)
        abc = ab2 * _safe_inv(proj_num, minval)
        proj_den = wp.dot(wp.cw_mul(abc, abc), vh2)
        dm = d_max[s]
        A_proj = wp.pi * dm * dm * wp.sqrt(proj_num * proj_den)

        normv = wp.cw_mul(abc, vh)
        cos_alpha = _safe_inv(proj_den, minval)
        circ = (c_kutta * rho * cos_alpha * A_proj) * wp.cross(normv, v)
        f = f + wp.cross(circ, v)

        IIs = II[s]
        Im = I_max[s]
        mom_visc = wp.cw_mul(w, c_ang * IIs + c_slender * (wp.vec3(Im, Im, Im) - IIs))
        drag_lin = mu * lin_visc_force[s] + rho * v_norm * (A_proj * c_blunt
                                                            + c_slender * (A_max[s] - A_proj))
        drag_ang = mu * lin_visc_torq[s] + rho * wp.length(mom_visc)
        f = f - drag_lin * v
        tau = tau - drag_ang * w

        if buoyancy != 0:
            g_s = RlsT * _qrot_inv(q, gravity)
            vol = volume_disp[s]
            cob = wp.vec3(0.0, 0.0, 0.0)
            if has_mesh[s] != 0:
                cob = mesh_centroid[s]
            if free_surface != 0:
                d = depth[e, s]
                frac = float(1.0)  # noqa: UP018
                if has_mesh[s] != 0:
                    R = mesh_radius[s]
                    if d >= R:
                        frac = 1.0
                        cob = mesh_centroid[s]
                    elif d <= -R:
                        frac = 0.0
                        cob = wp.vec3(0.0, 0.0, 0.0)
                    elif use_override != 0:
                        frac = frac_override[e, s]
                        cob = cob_override[e, s]
                else:
                    frac, cob = _cap(d, semi[s], up_s_in[e, s])
                if drag_scales != 0:
                    f = f * frac
                    tau = tau * frac
                vol = vol * frac
            f_b = (-rho * vol) * g_s
            tau = tau + wp.cross(cob, f_b)
            f = f + f_b

        f_l = Rls * f
        tau_l = Rls * tau + wp.cross(arm[ea, s], f_l)
        wp.atomic_add(F, e, b, f_l)
        wp.atomic_add(T, e, b, tau_l)


class WarpWrench:
    def __init__(self, hw):
        self.torch_device = hw.device
        self.device = wp.device_from_torch(hw.device)
        f32 = torch.float32

        def A(x, dtype):
            x = torch.as_tensor(x, device=hw.device).contiguous()
            return wp.from_torch(x.to(f32) if dtype is not wp.int32 else x.to(torch.int32),
                                 dtype=dtype)

        S = hw.n_slots
        self.S = S
        self.bidx = A(hw.bidx, wp.int32)
        self.slot_pos = A(hw.slot_pos, wp.vec3)
        self.R_ls = A(hw.R_ls, wp.mat33)
        self.semi = A(hw.semi, wp.vec3)
        self.vmass = A(hw.vmass, wp.vec3)
        self.vinert = A(hw.vinert, wp.vec3)
        self.coef = A(torch.stack([hw.c_blunt, hw.c_slender, hw.c_ang, hw.c_kutta,
                                   hw.c_magnus], dim=1), wp.float32)
        self.volume = A(hw.volume, wp.float32)
        self.d_max = A(hw.d_max, wp.float32)
        self.ab2_n = A(hw.ab2_n, wp.vec3)
        self.A_max = A(hw.A_max, wp.float32)
        self.I_max = A(hw.I_max, wp.float32)
        self.II = A(hw.II, wp.vec3)
        self.lin_visc_force = A(hw.lin_visc_force, wp.float32)
        self.lin_visc_torq = A(hw.lin_visc_torq, wp.float32)
        self.volume_disp = A(hw.volume_disp, wp.float32)

        md = hw.mesh_disp
        if md is not None:
            has = md.has_mesh.to(torch.int32)
            centroid, radius = md.centroid, md.radius
        else:
            has = torch.zeros(S, dtype=torch.int32, device=hw.device)
            centroid = torch.zeros(S, 3, device=hw.device)
            radius = torch.zeros(S, device=hw.device)
        self.has_mesh = A(has, wp.int32)
        self.mesh_centroid = A(centroid, wp.vec3)
        self.mesh_radius = A(radius, wp.float32)
        self.any_mesh = bool(has.any())

        self._n_cut = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.up = wp.vec3(*hw._up.detach().cpu().numpy().astype(np.float32).tolist())
        self._bufs = None
        wp.load_module(module=__name__, device=self.device)

    def _setup(self, hw, E, B):
        dev, S = self.torch_device, self.S

        def z(*shape):
            return torch.zeros(*shape, device=dev)

        t = {"quat": z(E, B, 4), "lin": z(E, B, 3), "ang": z(E, B, 3), "pos": z(E, B, 3),
             "cur": z(E, S, 3), "level": z(E, S), "depth": z(E, S), "up_s": z(E, S, 3),
             "frac_o": z(E, S), "cob_o": z(E, S, 3), "F": z(E, B, 3), "T": z(E, B, 3),
             "arm": z(1, S, 3)}
        vec = {4: wp.vec4, 3: wp.vec3}
        w = {k: wp.from_torch(v, dtype=vec[v.shape[-1]] if v.dim() == 3 else wp.float32)
             for k, v in t.items()}
        depth_cmd = wp.launch(_depth_kernel, dim=(E, S), device=self.device, record_cmd=True,
                              inputs=[w["quat"], w["pos"], self.bidx, self.slot_pos, self.R_ls,
                                      self.up, 0.0, w["level"], 0, self.mesh_radius,
                                      self.has_mesh, w["depth"], w["up_s"], self._n_cut])
        wrench_cmd = wp.launch(_wrench_kernel, dim=(E, S), device=self.device, record_cmd=True,
                               inputs=[
            w["quat"], w["lin"], w["ang"], w["cur"], wp.vec3(0.0, 0.0, 0.0), 0,
            self.bidx, self.slot_pos, self.R_ls, w["arm"], 0, 0,
            self.semi, self.vmass, self.vinert, self.coef, self.volume, self.d_max,
            self.ab2_n, self.A_max, self.I_max, self.II,
            self.lin_visc_force, self.lin_visc_torq,
            0.0, 0.0, float(hw.minval), 0, 0, 0,
            wp.vec3(0.0, 0.0, 0.0), self.up, self.volume_disp, w["depth"], w["up_s"],
            self.has_mesh, self.mesh_radius, self.mesh_centroid,
            0, w["frac_o"], w["cob_o"], w["F"], w["T"]])
        self._bufs = {"E": E, "B": B, "t": t, "w": w, "depth_cmd": depth_cmd,
                      "wrench_cmd": wrench_cmd, "params": {}, "arm_src": None}

    def _set(self, cmd_name, name, value):
        b = self._bufs
        key = (cmd_name, name)
        if b["params"].get(key, None) != value:
            b[cmd_name].set_param_by_name(name, value)
            b["params"][key] = value

    def compute(self, hw, quat_w, lin_vel_w, ang_vel_w, pos_w=None, current=None,
                surface=None, vel_at_com=False):
        E, B = quat_w.shape[0], quat_w.shape[1]
        if B != hw.num_bodies:
            raise ValueError(f"expected {hw.num_bodies} bodies, got {B}")
        if self._bufs is None or self._bufs["E"] != E or self._bufs["B"] != B:
            self._setup(hw, E, B)
        b = self._bufs
        t, w = b["t"], b["w"]
        S = self.S

        free_surface = hw.free_surface or surface is not None
        need_pos = (hw.buoyancy and free_surface) or callable(current)
        if need_pos and pos_w is None:
            raise ValueError("pos_w is required for a free surface or a current field")

        t["quat"].copy_(quat_w)
        t["lin"].copy_(lin_vel_w)
        t["ang"].copy_(ang_vel_w)
        if pos_w is not None and need_pos:
            t["pos"].copy_(pos_w)

        centre_w = None
        if callable(current) or callable(surface):
            q = quat_w[:, hw.bidx]
            u = q[..., 1:4]
            p = hw.slot_pos.expand(E, -1, -1)
            tt = torch.cross(u, p, dim=-1)
            centre_w = pos_w[:, hw.bidx] + p + 2.0 * (q[..., 0:1] * tt
                                                      + torch.cross(u, tt, dim=-1))

        cur_const, use_cur = (0.0, 0.0, 0.0), 0
        if callable(current):
            t["cur"].copy_(torch.as_tensor(current(centre_w), dtype=torch.float32,
                                           device=quat_w.device).expand(E, S, 3))
            use_cur = 2
        elif current is not None:
            c = torch.as_tensor(current, dtype=torch.float32, device=quat_w.device)
            if c.numel() == 3:
                cur_const, use_cur = tuple(c.reshape(3).tolist()), 1
            else:
                t["cur"].copy_(c.expand(E, S, 3))
                use_cur = 2
        elif hw._has_current:
            cur_const = tuple(np.asarray(hw.medium.current, dtype=np.float32).tolist())
            use_cur = 1
        self._set("wrench_cmd", "current_const", wp.vec3(*cur_const))
        self._set("wrench_cmd", "use_current", use_cur)
        self._set("wrench_cmd", "vel_at_com", int(vel_at_com))

        if hw.arm is not b["arm_src"]:
            arm = hw.arm.to(torch.float32)
            if arm.shape[0] != t["arm"].shape[0]:
                t["arm"] = arm.contiguous().clone()
                w["arm"] = wp.from_torch(t["arm"], dtype=wp.vec3)
                b["wrench_cmd"].set_param_by_name("arm", w["arm"])
            else:
                t["arm"].copy_(arm)
            self._set("wrench_cmd", "arm_per_env", int(arm.shape[0] > 1))
            b["arm_src"] = hw.arm

        self._set("wrench_cmd", "rho", float(hw.medium.density))
        self._set("wrench_cmd", "mu", float(hw.medium.viscosity))
        self._set("wrench_cmd", "buoyancy", int(hw.buoyancy))
        self._set("wrench_cmd", "free_surface", int(free_surface))
        self._set("wrench_cmd", "drag_scales", int(hw.drag_scales_with_submersion))
        self._set("wrench_cmd", "gravity",
                  wp.vec3(*np.asarray(hw.medium.gravity, dtype=np.float32).tolist()))

        stream = None
        if self.device.is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(quat_w.device))
        with wp.ScopedStream(stream) if stream is not None else nullcontext():
            use_override = 0
            if hw.buoyancy and free_surface:
                use_level = 0
                if surface is not None:
                    t["level"].copy_(torch.as_tensor(surface(centre_w), dtype=torch.float32,
                                                     device=quat_w.device).expand(E, S))
                    use_level = 1
                self._set("depth_cmd", "surface_z", float(hw._surface_z))
                self._set("depth_cmd", "use_level", use_level)
                if self.any_mesh:
                    self._n_cut.zero_()
                b["depth_cmd"].launch(stream)
                if self.any_mesh and int(self._n_cut.numpy()[0]) > 0:
                    m_frac, m_cob = hw.mesh_disp.submerged(t["depth"], t["up_s"])
                    t["frac_o"].copy_(m_frac)
                    t["cob_o"].copy_(m_cob)
                    use_override = 1
            self._set("wrench_cmd", "use_override", use_override)
            t["F"].zero_()
            t["T"].zero_()
            b["wrench_cmd"].launch(stream)
        return t["F"].clone(), t["T"].clone()

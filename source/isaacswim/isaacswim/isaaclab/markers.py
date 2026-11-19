from __future__ import annotations

import math

import numpy as np

from isaacswim.isaaclab import _lab
from isaacswim.np_math import quat_apply as _quat_rot
from isaacswim.np_math import quat_from_to as _quat_from_to
from isaacswim.np_math import quat_mul as _quat_mul

__all__ = ["ThrusterVisuals", "CurrentVisuals"]


ARROW_AXIS = np.array([1.0, 0.0, 0.0])


class ThrusterVisuals:
    def __init__(
        self,
        thr,
        robot,
        *,
        prim_root: str = "/World/rov",
        marker_path="/Visuals/thrust_arrows",
        body: str = "base_link",
        ref_thrust: float | None = None,
        max_len: float = 0.34,
        scale: str = "linear",
        knee: float = 1.0,
        girth: float = 0.07,
        standoff: float = 0.075,
        max_rev_per_s: float = 4.0,
    ):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

        self.thr = getattr(thr, "array", thr)
        self.body = body
        self.robot = robot
        if scale not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', not {scale!r}")
        self.scale = scale
        self.knee = max(1e-6, float(knee))
        if ref_thrust is None:
            ref_thrust = float(np.max(self.thr.fwd)) if scale == "log" else 2.5
        self.ref_thrust = max(1e-6, float(ref_thrust))
        self.max_len = float(max_len)
        self.girth = float(girth)
        self.standoff = float(standoff)
        self.max_w = 2.0 * math.pi * float(max_rev_per_s)
        self.n = self.thr.n
        self._angle = np.zeros(self.n)

        self.arrows = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=marker_path,
                markers={
                    "thrust": GREEN_ARROW_X_MARKER_CFG.markers["arrow"].replace(
                        scale=(1.0, 1.0, 1.0)
                    )
                },
            )
        )

        self._bind_props(prim_root)

    def _bind_props(self, prim_root: str):
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        self._ops, self._base_q, self._spin_axis = [], [], []
        for i, name in enumerate(self.thr.names):
            prim = stage.GetPrimAtPath(f"{prim_root}/{self.body}/{name}")
            if not prim or not prim.IsValid():
                self._ops.append(None)
                self._base_q.append(None)
                self._spin_axis.append(None)
                continue
            xf = UsdGeom.Xformable(prim)
            op = next(
                (o for o in xf.GetOrderedXformOps() if o.GetOpName() == "xformOp:orient"), None
            )
            if op is None:
                op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            q = op.Get()
            base = np.array([q.GetReal(), *[float(c) for c in q.GetImaginary()]])
            local = _quat_rot(np.array([base[0], -base[1], -base[2], -base[3]]), self.thr.axis[i])
            self._ops.append(op)
            self._base_q.append(base)
            self._spin_axis.append(local / (np.linalg.norm(local) + 1e-12))

    def _fraction(self, mag: float) -> float:
        if self.scale == "log":
            return math.log1p(mag / self.knee) / math.log1p(self.ref_thrust / self.knee)
        return mag / self.ref_thrust

    def update(self, thrust, dt: float):
        import torch
        from pxr import Gf

        t = np.asarray(thrust, dtype=np.float64).reshape(self.n)
        root_p = _lab.data(self.robot).root_pos_w[0].detach().cpu().numpy()
        root_q = _lab.data(self.robot).root_quat_w[0].detach().cpu().numpy()

        pos = np.zeros((self.n, 3))
        quats = np.zeros((self.n, 4))
        scales = np.ones((self.n, 3))
        idx = np.zeros(self.n, dtype=np.int64)

        push_b = self.thr.A[:3, :].T

        for i in range(self.n):
            mag = abs(float(t[i]))
            frac = min(1.0, self._fraction(mag))
            sign = 1.0 if t[i] >= 0.0 else -1.0
            d_w = _quat_rot(root_q, -push_b[i] * sign)
            nrm = np.linalg.norm(d_w)
            d_w = d_w / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])

            pos[i] = root_p + _quat_rot(root_q, self.thr.pos[i]) + d_w * self.standoff
            quats[i] = _quat_from_to(ARROW_AXIS, d_w)
            scales[i] = (max(self.max_len * frac, 1e-3), self.girth, self.girth)

            op = self._ops[i]
            if op is not None:
                self._angle[i] += self.max_w * frac * sign * dt
                a = self._angle[i]
                ax = self._spin_axis[i]
                s2 = math.sin(a / 2.0)
                spin = np.array([math.cos(a / 2.0), ax[0] * s2, ax[1] * s2, ax[2] * s2])
                q_new = _quat_mul(self._base_q[i], spin)
                op.Set(
                    Gf.Quatd(
                        float(q_new[0]), Gf.Vec3d(float(q_new[1]), float(q_new[2]), float(q_new[3]))
                    )
                )

        dev = self.robot.device
        quats = torch.as_tensor(quats, device=dev, dtype=torch.float32)
        if _lab.is_lab3(self.robot):
            quats = _lab.quat_to_lab(quats)
        self.arrows.visualize(
            translations=torch.as_tensor(pos, device=dev, dtype=torch.float32),
            orientations=quats,
            scales=torch.as_tensor(scales, device=dev, dtype=torch.float32),
            marker_indices=torch.as_tensor(idx, device=dev),
        )


class CurrentVisuals:
    def __init__(
        self,
        gusts,
        robot,
        *,
        seabed_z: float = -0.55,
        n: int = 420,
        box=(2.6, 2.6, 1.15),
        radius: float = 0.0015,
        seed: int = 3,
        swirl=((1.20, 0.25), (0.55, 0.12), (0.25, 0.06)),
        jitter: float = 0.12,
        jitter_tau: float = 0.9,
        emit: float = 0.5,
        marker_path="/Visuals/current",
    ):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.gusts = gusts
        self.robot = robot
        self.seabed_z = float(seabed_z)
        self.box = np.asarray(box, dtype=float)
        self.n = int(n)

        rng = self._rng = np.random.default_rng(seed)
        self.p: np.ndarray | None = None

        self.modes = [(2.0 * np.pi / L, amp / (2.0 * np.pi / L)) for L, amp in swirl]
        self._src = [f for f in getattr(gusts, "fields", []) if hasattr(f, "tracer_seed")]
        self.emit = float(emit) if self._src else 0.0
        self.jitter = float(jitter)
        self.jitter_tau = float(jitter_tau)
        self.j = np.zeros((self.n, 3))
        self.t = 0.0
        self.scales = np.repeat(rng.uniform(0.55, 1.5, (self.n, 1)), 3, axis=1)

        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=marker_path,
                markers={
                    "bubble": sim_utils.SphereCfg(
                        radius=float(radius),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.72, 0.90, 0.98), roughness=0.25
                        ),
                    )
                },
            )
        )

    def reset(self, gusts=None) -> None:
        if gusts is not None:
            self.gusts = gusts
            self._src = [f for f in getattr(gusts, "fields", []) if hasattr(f, "tracer_seed")]
        self.p = None
        self.j[:] = 0.0

    def _bounds(self, root):
        half = self.box * 0.5
        lo = np.array(
            [root[0] - half[0], root[1] - half[1], max(self.seabed_z + 0.02, root[2] - half[2])]
        )
        return lo, lo + self.box

    def _swirl(self, world, speed):
        x, y, z = world[:, 0], world[:, 1], world[:, 2]
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        w = np.zeros_like(x)
        for i, (k, a) in enumerate(self.modes):
            ph = 1.7 * i
            px, py = k * x + ph + 0.30 * self.t, k * y - ph + 0.21 * self.t
            u += a * k * np.sin(px) * np.cos(py)
            v += -a * k * np.cos(px) * np.sin(py)
            w += 0.5 * a * k * np.sin(1.7 * k * z + ph + 0.5 * self.t)
        return np.column_stack([u, v, w]) * speed[:, None]

    def update(self, dt: float = 1.0 / 30.0):
        import torch

        dev = self.robot.device
        root = _lab.data(self.robot).root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        lo, hi = self._bounds(root)
        if self.p is None:
            self.p = lo + self._rng.random((self.n, 3)) * self.box
        world = self.p
        v = self.gusts(torch.as_tensor(world, device=dev, dtype=torch.float32).unsqueeze(0))
        flow = v[0].detach().cpu().numpy()

        self.t += dt
        speed = np.linalg.norm(flow, axis=1)
        a = np.exp(-dt / self.jitter_tau)
        self.j = a * self.j + np.sqrt(1.0 - a * a) * self._rng.standard_normal((self.n, 3))
        self.p += (flow + self._swirl(world, speed) + self.j * (self.jitter * speed)[:, None]) * dt

        gone = ((self.p < lo) | (self.p > hi)).any(axis=1)
        self.p = lo + np.mod(self.p - lo, self.box)
        if self.emit:
            born = gone & (self._rng.random(self.n) < self.emit)
            if born.any():
                f = self._src[self._rng.integers(len(self._src))]
                self.p[born] = f.tracer_seed(int(born.sum()), self._rng)

        self.markers.visualize(
            translations=torch.as_tensor(self.p, device=dev, dtype=torch.float32),
            scales=torch.as_tensor(self.scales, device=dev, dtype=torch.float32),
        )

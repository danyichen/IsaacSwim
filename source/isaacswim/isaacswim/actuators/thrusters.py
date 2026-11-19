from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import torch

from isaacswim.actuators.base import Actuator
from isaacswim.actuators.motor import Motor
from isaacswim.actuators.propeller import Propeller, advance_speed

_MAX_EDGES = 5000


def _edge_tables(null, fwd: float, rev: float):
    from itertools import combinations

    n, k = null.shape
    G = np.vstack([np.eye(n), -np.eye(n)])
    c = np.concatenate([np.full(n, fwd), np.full(n, rev)])
    GN = G @ null
    if math.comb(2 * n, k) > _MAX_EDGES:
        return None
    L, beta, Pc, PG = [], [], [], []
    for idx in combinations(range(2 * n), k):
        idx = list(idx)
        M = GN[idx]
        if k and abs(np.linalg.det(M)) < 1e-9:
            continue
        P = np.linalg.inv(M) if k else np.zeros((0, 0))
        L.append(G - GN @ P @ G[idx])
        beta.append(c - GN @ P @ c[idx])
        Pc.append(P @ c[idx])
        PG.append(P @ G[idx])
    return tuple(torch.as_tensor(np.asarray(x), dtype=torch.float64)
                 for x in (L, beta, Pc, PG))


def _max_scale(b, null, tables, lo: float, hi: float):
    L, beta, Pc, PG = tables
    alpha = torch.einsum("sjn,en->esj", L, b)
    tol = 1e-12
    ratio = beta / torch.where(alpha.abs() > tol, alpha, torch.ones_like(alpha))
    inf = torch.full_like(ratio, math.inf)
    s_hi = torch.where(alpha > tol, ratio, inf).amin(-1).clamp(max=1.0)
    s_lo = torch.where(alpha < -tol, ratio, -inf).amax(-1).clamp(min=0.0)
    ok = (s_lo <= s_hi + 1e-12) & ((alpha.abs() > tol) | (beta >= -1e-9)).all(-1)
    s = torch.where(ok, s_hi, torch.full_like(s_hi, -1.0)).amax(-1).clamp(min=0.0)
    z = Pc[None] - s[:, None, None] * torch.einsum("skn,en->esk", PG, b)
    reach = ok & (s_hi >= s[:, None] - 1e-9) & (s_lo <= s[:, None] + 1e-9)
    edge = torch.where(reach, z.square().sum(-1), inf[..., 0]).argmin(-1)
    z = z[torch.arange(len(s), device=b.device), edge]
    return s, (s[:, None] * b + z @ null.T).clamp(lo, hi)

__all__ = ["ThrusterArray", "Thrusters", "T200_FWD_N", "T200_REV_N", "WRENCH_NAMES"]

T200_FWD_N = 5.25 * 9.80665
T200_REV_N = 4.10 * 9.80665

WRENCH_NAMES = ("fx", "fy", "fz", "mx", "my", "mz")


class ThrusterArray:
    def __init__(self, json_path=None, *, sign=1.0, fwd: float = T200_FWD_N,
                 rev: float = T200_REV_N, _geometry=None):
        if _geometry is None:
            if json_path is None:
                raise TypeError("give a thrusters JSON path, or use "
                                "ThrusterArray.from_geometry")
            data = json.loads(pathlib.Path(json_path).read_text())
            frame = data["frame"]
            names = [t["link"] for t in data["thrusters"]]
            pos = [t["position"] for t in data["thrusters"]]
            axis = [t["axis"] for t in data["thrusters"]]
        else:
            frame, names, pos, axis = _geometry

        self.frame = frame
        self.names = list(names)
        self.pos = np.array(pos, dtype=np.float64).reshape(-1, 3)
        self.axis = np.array(axis, dtype=np.float64).reshape(-1, 3)
        if len(self.names) != len(self.pos) or len(self.pos) != len(self.axis):
            raise ValueError("names, positions and axes must have one entry per thruster")
        if not self.names or not np.isfinite(self.pos).all() or not np.isfinite(self.axis).all():
            raise ValueError("thruster geometry must be nonempty and finite")
        if np.any(np.linalg.norm(self.axis, axis=1) == 0):
            raise ValueError("thruster axes must be nonzero")
        self.axis /= np.linalg.norm(self.axis, axis=1, keepdims=True)

        self.n = len(self.names)
        self.sign = np.broadcast_to(np.asarray(sign, dtype=np.float64), (self.n,)).copy()
        self.fwd = float(fwd)
        self.rev = float(rev)
        if not (np.isfinite([self.fwd, self.rev]).all() and self.fwd > 0 and self.rev > 0):
            raise ValueError("thrust limits must be finite and positive")
        if not np.isin(self.sign, [-1.0, 1.0]).all():
            raise ValueError("thruster signs must be +1 or -1")

        self.com = np.zeros(3)
        self._build()

    @classmethod
    def from_geometry(cls, positions, axes, *, names: Sequence[str] | None = None,
                      frame: str = "base_link", **kw) -> ThrusterArray:
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if names is None:
            names = [f"thruster{i}" for i in range(len(positions))]
        return cls(_geometry=(frame, names, positions, axes), **kw)

    def _build(self):
        d = self.axis * self.sign[:, None]
        r = self.pos - self.com
        self.A = np.zeros((6, self.n))
        self.A[:3, :] = d.T
        self.A[3:, :] = np.cross(r, d).T
        self.pinv = np.linalg.pinv(self.A)
        _, s, vt = np.linalg.svd(self.A)
        tol = max(self.A.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
        self.null = vt[int((s > tol).sum()):].T

    def rebuild_about_com(self, com):
        self.com = np.asarray(com, dtype=np.float64).reshape(3)
        self._build()
        return self

    def set_sign(self, sign):
        self.sign = np.broadcast_to(np.asarray(sign, dtype=np.float64), (self.n,)).copy()
        self._build()
        return self

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    @property
    def cond(self) -> float:
        return float(np.linalg.cond(self.A))

    def wrench_of(self, thrust) -> np.ndarray:
        return self.A @ np.asarray(thrust, dtype=np.float64)

    def max_wrench_axis(self, k: int) -> float:
        lo, hi = 0.0, 1.0
        while self._feasible(self._unit(k) * hi) and hi < 1e4:
            lo, hi = hi, hi * 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._feasible(self._unit(k) * mid):
                lo = mid
            else:
                hi = mid
        return lo

    @staticmethod
    def _unit(k):
        e = np.zeros(6)
        e[k] = 1.0
        return e

    def _feasible(self, wrench, iters: int = 200) -> bool:
        _, ok = self._project(np.asarray(wrench, dtype=np.float64), iters)
        return ok

    def _clip(self, t):
        return np.clip(t, -self.rev, self.fwd)

    def _inside(self, t, tol=1e-6) -> bool:
        return bool(np.all(t <= self.fwd + tol) and np.all(t >= -self.rev - tol))

    def _project(self, wrench, iters: int):
        base = self.pinv @ wrench
        if not np.allclose(self.A @ base, wrench, rtol=1e-9, atol=1e-9):
            return np.zeros(self.n), False
        t = base.copy()
        if self._inside(t, 1e-9):
            return t, True
        for _ in range(iters):
            t_box = self._clip(t)
            t = base + self.null @ (self.null.T @ (t_box - base))
            if self._inside(t):
                return self._clip(t), True
        return self._clip(t), False

    def allocate(self, wrench, *, iters: int = 200):
        w = np.asarray(wrench, dtype=np.float64).reshape(6)
        if not np.isfinite(w).all():
            raise ValueError("wrench must be finite")
        tables = self.edge_tables()
        base = self.pinv @ w
        if not np.allclose(self.A @ base, w, rtol=1e-9, atol=1e-9):
            t = np.zeros(self.n)
            return t, {"scale": 0.0, "realised": self.A @ t, "saturated": True}
        if tables is not None and not self._inside(base, 1e-9):
            s, t = _max_scale(torch.as_tensor(base)[None], torch.as_tensor(self.null),
                              tables, -self.rev, self.fwd)
            if float(s[0]) < 1.0 - 1e-9:
                t = t[0].numpy()
                return t, {"scale": float(s[0]), "realised": self.A @ t, "saturated": True}
        t, ok = self._project(w, iters)
        scale = 1.0
        if not ok and tables is None:
            lo, hi = 0.0, 1.0
            t = np.zeros(self.n)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                t_mid, ok_mid = self._project(w * mid, iters)
                if ok_mid:
                    lo, t = mid, t_mid
                else:
                    hi = mid
            scale = lo
        return t, {"scale": scale, "realised": self.A @ t,
                   "saturated": not ok and tables is None}

    def edge_tables(self):
        key = (self.fwd, self.rev, self.null.tobytes())
        if getattr(self, "_edges_key", None) != key:
            self._edges = _edge_tables(self.null, self.fwd, self.rev)
            self._edges_key = key
        return self._edges

    def body_wrench(self, thrust):
        w = self.A @ np.asarray(thrust, dtype=np.float64)
        return w[:3], w[3:]

    def __repr__(self) -> str:
        return (f"ThrusterArray({self.n} thrusters, rank {self.rank}, "
                f"cond {self.cond:.1f}, null dim {self.null.shape[1]}, "
                f"limits +{self.fwd:.1f}/-{self.rev:.1f} N)")


class Thrusters(Actuator):
    def __init__(self, array: ThrusterArray, *, body: str = "base_link",
                 name: str = "thrusters", input: str = "wrench",
                 allocation: str = "nullspace", time_constant: float = 0.0,
                 propeller: Propeller | None = None, motor: Motor | None = None,
                 rho: float = 1025.0):
        super().__init__(name)
        if input not in ("wrench", "thrust"):
            raise ValueError(f"input must be 'wrench' or 'thrust', not {input!r}")
        if allocation not in ("nullspace", "pinv"):
            raise ValueError(f"allocation must be 'nullspace' or 'pinv', not {allocation!r}")
        self.array = array
        self.body = body
        self.input = input
        self.allocation = allocation
        self.time_constant = float(time_constant)
        self.propeller = propeller
        self.rho = float(rho)
        self.motor = motor
        self.needs_state = propeller is not None
        if motor is not None:
            if propeller is None or propeller.kq0_fwd is None:
                raise ValueError("a motor needs a propeller with torque coefficients")
            array.fwd, array.rev = self._full_duty_thrust()
        elif propeller is not None:
            array.fwd, array.rev = propeller.max_thrust(self.rho)
        self.size = 6 if input == "wrench" else array.n
        self.body_index = -1
        self.info: dict = {}

    def bind(self, *, body_names, joint_names, num_envs, device, body_com=None,
             default_joint_pos=None) -> None:
        super().bind(body_names=body_names, joint_names=joint_names,
                     num_envs=num_envs, device=device)
        names = list(body_names)
        if self.body not in names:
            raise ValueError(f"{self.name}: body {self.body!r} is not in the "
                             f"articulation; it has {names}")
        self.body_index = names.index(self.body)
        if body_com is not None:
            com = np.asarray(torch.as_tensor(body_com).detach().cpu(), dtype=np.float64)
            self.array.rebuild_about_com(com.reshape(-1, 3)[self.body_index])
        self._sync_geometry()
        n = self.array.n
        self.command = torch.zeros(num_envs, n, dtype=torch.float64, device=self.device)
        self.thrust = torch.zeros_like(self.command)
        self.rps = torch.zeros_like(self.command)
        self.duty = torch.zeros_like(self.command)
        self.current = torch.zeros_like(self.command)
        self.info = {"scale": torch.ones(num_envs, dtype=torch.float64),
                     "saturated": torch.zeros(num_envs, dtype=torch.bool)}

    def _sync_geometry(self) -> None:
        self._full_rank = self.array.rank == 6
        self._A = torch.as_tensor(self.array.A, dtype=torch.float64, device=self.device)
        self._pinv = torch.as_tensor(self.array.pinv, dtype=torch.float64,
                                     device=self.device)
        self._null = torch.as_tensor(self.array.null, dtype=torch.float64,
                                     device=self.device)
        self._pos = torch.as_tensor(self.array.pos, dtype=torch.float64,
                                    device=self.device)
        self._push = torch.as_tensor(self.array.axis * self.array.sign[:, None],
                                     dtype=torch.float64, device=self.device)
        tables = self.array.edge_tables()
        self._edges = None if tables is None else tuple(x.to(self.device) for x in tables)

    def reset(self, env_ids=None) -> None:
        if not self._bound:
            return
        ids = slice(None) if env_ids is None else env_ids
        self.command[ids] = 0.0
        self.thrust[ids] = 0.0
        self.rps[ids] = 0.0
        self.duty[ids] = 0.0
        self.current[ids] = 0.0

    def set_setpoint(self, value) -> None:
        self._require_bound()
        lo, hi = -self.array.rev, self.array.fwd
        if self.input == "thrust":
            t = self._as_batch(value, self.array.n, torch.float64)
            self.command = t.clamp(lo, hi)
            return

        w = self._as_batch(value, 6, torch.float64)
        if self.allocation == "pinv":
            t = w @ self._pinv.T
            clipped = t.clamp(lo, hi)
            self.info = {"scale": torch.ones(self.num_envs, dtype=torch.float64),
                         "saturated": (clipped != t).any(dim=1).cpu()}
            self.command = clipped
            return

        out, scale, sat = self._allocate_batch(w)
        self.info = {"scale": scale.cpu(), "saturated": sat.cpu()}
        self.command = out

    def _project_batch(self, w, iters: int = 200):
        lo, hi = -self.array.rev, self.array.fwd

        def inside(t, tol):
            return ((t <= hi + tol) & (t >= lo - tol)).all(dim=1)

        base = w @ self._pinv.T
        t = base.clone()
        out = base.clone()
        done = inside(t, 1e-9)
        ok = done.clone()
        for i in range(iters):
            if i % 8 == 0 and bool(done.all()):
                break
            t_box = t.clamp(lo, hi)
            t_new = base + ((t_box - base) @ self._null) @ self._null.T
            ins = inside(t_new, 1e-6) & ~done
            out = torch.where(ins.unsqueeze(1), t_new.clamp(lo, hi), out)
            ok = ok | ins
            done = done | ins
            t = torch.where(done.unsqueeze(1), t, t_new)
        out = torch.where(done.unsqueeze(1), out, t.clamp(lo, hi))
        return out, ok

    def _allocate_batch(self, w, iters: int = 200):
        if not self._full_rank:
            projected = (w @ self._pinv.T) @ self._A.T
            reachable = torch.isclose(projected, w, rtol=1e-9, atol=1e-9).all(dim=1)
            if not bool(reachable.all()):
                t = torch.zeros(w.shape[0], self.array.n, dtype=w.dtype, device=w.device)
                scale = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
                sat = ~reachable
                if bool(reachable.any()):
                    allocated = self._allocate_batch(w[reachable])
                    t[reachable], scale[reachable], sat[reachable] = allocated
                return t, scale, sat
        scale = torch.ones(w.shape[0], dtype=w.dtype, device=w.device)
        if self._edges is not None:
            lo_t, hi_t = -self.array.rev, self.array.fwd
            base = w @ self._pinv.T
            out = ~((base <= hi_t + 1e-9) & (base >= lo_t - 1e-9)).all(dim=1)
            sat = torch.zeros_like(out)
            t = torch.zeros_like(base)
            if bool(out.any()):
                s, tb = _max_scale(base[out], self._null, self._edges, lo_t, hi_t)
                scale[out], t[out] = s, tb
                sat[out] = s < 1.0 - 1e-9
                scale[out & ~sat] = 1.0
            if bool((~sat).any()):
                t[~sat], _ = self._project_batch(w[~sat], iters)
            return t, scale, sat
        t, ok = self._project_batch(w, iters)
        bad = ~ok
        if bool(bad.any()):
            wb = w[bad]
            lo = torch.zeros(wb.shape[0], dtype=w.dtype, device=w.device)
            hi = torch.ones_like(lo)
            tb = torch.zeros(wb.shape[0], self.array.n, dtype=w.dtype, device=w.device)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                tm, okm = self._project_batch(wb * mid.unsqueeze(1), iters)
                lo = torch.where(okm, mid, lo)
                hi = torch.where(okm, hi, mid)
                tb = torch.where(okm.unsqueeze(1), tm, tb)
            t = t.clone()
            t[bad] = tb
            scale[bad] = lo
        return t, scale, bad

    def wrench(self, dt: float, state=None):
        self._require_bound()
        a = 1.0 - math.exp(-dt / self.time_constant) if self.time_constant > 0.0 else 1.0
        if self.propeller is not None:
            if state is None:
                u = torch.zeros_like(self.rps)
            else:
                f64 = dict(dtype=torch.float64, device=self.device)
                cur = None if state.current_b is None else state.current_b.to(**f64)
                u = advance_speed(state.lin_vel_b.to(**f64), state.ang_vel_b.to(**f64),
                                  cur, self._pos, self._push)
            if self.motor is None:
                n_cmd = self.propeller.rps_for_thrust(self.command, self.rho)
                self.rps = self.rps + a * (n_cmd - self.rps)
            else:
                self._spin(u, a, dt)
            self.thrust = self.propeller.thrust(self.rps, u, self.rho)
        else:
            self.thrust = self.thrust + a * (self.command - self.thrust)
        w = self.thrust @ self._A.T
        return self.body_index, w[:, :3], w[:, 3:]

    def _spin(self, u: torch.Tensor, a: float, dt: float) -> None:
        prop, motor = self.propeller, self.motor
        n_cmd = prop.rps_for_thrust(self.command, self.rho, clamp=False)
        q_cmd, _ = prop.torque(n_cmd, torch.zeros_like(n_cmd), self.rho)
        d_cmd = motor.duty_for_speed(2.0 * math.pi * n_cmd, q_cmd).clamp(-1.0, 1.0)
        self.duty = self.duty + a * (d_cmd - self.duty)
        w, i = motor.step(2.0 * math.pi * self.rps, self.duty,
                          lambda w: prop.torque(w / (2.0 * math.pi), u, self.rho), dt)
        self.rps = w / (2.0 * math.pi)
        self.current = self.duty * i

    def _full_duty_thrust(self) -> tuple[float, float]:
        prop, motor = self.propeller, self.motor
        out = []
        for kt0, kq0 in ((prop.kt0_fwd, prop.kq0_fwd), (prop.kt0_rev, prop.kq0_rev)):
            c = self.rho * prop.diameter ** 5 * kq0 / (2.0 * math.pi) ** 2
            n = motor.bollard_speed(1.0, c) / (2.0 * math.pi)
            out.append(self.rho * prop.diameter ** 4 * kt0 * n * n)
        return out[0], out[1]

    @property
    def power(self) -> torch.Tensor:
        return self.motor.voltage * self.current.sum(dim=1)

    def __repr__(self) -> str:
        return (f"Thrusters({self.name!r}, {self.array.n} on {self.body!r}, "
                f"input={self.input}, allocation={self.allocation})")

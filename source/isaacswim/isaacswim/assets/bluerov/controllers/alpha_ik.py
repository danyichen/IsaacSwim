from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from isaacswim.np_math import rpy_to_mat as _rpy_to_mat

__all__ = ["AlphaIK"]

ARM_JOINTS = ("alpha_axis_e", "alpha_axis_d", "alpha_axis_c", "alpha_axis_b")


def _axis_angle(axis, q):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    k = axis / n
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(q) * K + (1.0 - math.cos(q)) * (K @ K)


class AlphaIK:
    def __init__(self, urdf_path, *, tool_offset: float = 0.085,
                 tip_link: str = "alpha_standard_jaws_rs1_130_link",
                 mount_joint: str = "alpha_jaws_joint"):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0")
                                      if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0")
                                      if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz") if ax is not None
                                       else "0 0 1").split()]
            lim = j.find("limit")
            self.joints[j.get("name")] = {
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "t": np.array(xyz), "R": _rpy_to_mat(rpy), "axis": np.array(axis),
                "lower": float(lim.get("lower")) if lim is not None and
                lim.get("lower") is not None else None,
                "upper": float(lim.get("upper")) if lim is not None and
                lim.get("upper") is not None else None,
            }

        parent_of = {v["child"]: k for k, v in self.joints.items()}
        chain, node = [], tip_link
        while node in parent_of:
            jn = parent_of[node]
            chain.append(jn)
            node = self.joints[jn]["parent"]
        chain.reverse()
        self.chain = chain
        if node != "base_link":
            raise ValueError(f"chain from {tip_link} ends at {node}, not base_link")

        self.limits = {n: (self.joints[n]["lower"], self.joints[n]["upper"])
                       for n in ARM_JOINTS}
        self.tool_offset = float(tool_offset)
        if mount_joint not in self.joints:
            raise ValueError(f"{mount_joint} is not a joint in {urdf_path}; the "
                             f"tool mount is what `tool_offset` is measured from")
        self.mount_joint = mount_joint

        self._build_planar()

    def _fixed_prefix(self):
        R, t = np.eye(3), np.zeros(3)
        for jn in self.chain:
            j = self.joints[jn]
            t = t + R @ j["t"]
            R = R @ j["R"]
            if jn == "alpha_axis_e":
                return R, t
        raise ValueError("alpha_axis_e not on the chain")

    def _build_planar(self):
        j_d = self.joints["alpha_axis_d"]
        j_c = self.joints["alpha_axis_c"]
        j_b = self.joints["alpha_axis_b"]
        j_mount = self.joints[self.mount_joint]

        self.R_arm, self.t_arm = self._fixed_prefix()

        w = j_b["t"] + j_mount["t"] + j_mount["R"] @ np.array(
            [0.0, 0.0, self.tool_offset])

        for name, v in (("t_d", j_d["t"]), ("t_c", j_c["t"]), ("w", w)):
            if abs(float(v[1])) > 1e-9:
                raise ValueError(f"{name} has a y component ({v[1]}); the arm is not "
                                 "planar and this closed form does not apply")

        self.t_d = complex(j_d["t"][0], j_d["t"][2])
        self.t_c = complex(j_c["t"][0], j_c["t"][2])
        self.w = complex(w[0], w[2])
        if not np.allclose(j_c["R"], _rpy_to_mat([0, 0, math.pi]), atol=1e-4):
            raise ValueError("alpha_axis_c's origin rotation is not R_z(pi); the "
                             "reflection assumed by this closed form has changed")
        self.W = -np.conj(self.w)

        self.L1 = abs(self.t_c)
        self.L2 = abs(self.W)
        self.delta = np.angle(self.t_c) - np.angle(self.W)

    def fk(self, q: dict) -> tuple[np.ndarray, np.ndarray]:
        R, t = np.eye(3), np.zeros(3)
        for jn in self.chain:
            j = self.joints[jn]
            t = t + R @ j["t"]
            R = R @ j["R"]
            if j["type"] in ("revolute", "continuous"):
                R = R @ _axis_angle(j["axis"], float(q.get(jn, 0.0)))
        j_fin = self.joints[self.chain[-1]]
        t = t - R @ j_fin["t"]
        return t + R @ np.array([0.0, 0.0, self.tool_offset]), R

    def reach(self) -> tuple[float, float]:
        return abs(self.L1 - self.L2), self.L1 + self.L2

    def _in_limits(self, name, q, tol=1e-6):
        lo, hi = self.limits[name]
        return (lo is None or q >= lo - tol) and (hi is None or q <= hi + tol)

    @staticmethod
    def _wrap_into(q, lo, hi):
        for k in (-1, 0, 1):
            v = q + k * 2.0 * math.pi
            if lo - 1e-6 <= v <= hi + 1e-6:
                return v
        return q

    def ik(self, target_base, *, roll: float | None = None, seed: dict | None = None):
        p = np.asarray(target_base, dtype=np.float64).reshape(3)
        seed = dict(seed or {})

        p_a = self.R_arm.T @ (p - self.t_arm)

        radial = math.hypot(p_a[0], p_a[1])
        yaw0 = math.atan2(p_a[1], p_a[0])

        best, best_cost, reason = None, float("inf"), "unreachable"
        lo_e, hi_e = self.limits["alpha_axis_e"]

        for yaw_branch, x1 in ((yaw0, radial), (yaw0 + math.pi, -radial)):
            q_e = self._wrap_into(yaw_branch, lo_e, hi_e)
            if not self._in_limits("alpha_axis_e", q_e):
                reason = "yaw outside axis_e limits"
                continue

            u = complex(x1, p_a[2]) - self.t_d
            r = abs(u)
            if r > self.L1 + self.L2 + 1e-9 or r < abs(self.L1 - self.L2) - 1e-9:
                continue
            cos_arg = (r * r - self.L1 ** 2 - self.L2 ** 2) / (2.0 * self.L1 * self.L2)
            cos_arg = max(-1.0, min(1.0, cos_arg))
            base = math.acos(cos_arg)

            for elbow in (+1.0, -1.0):
                q_c = self.delta - elbow * base
                q_c = self._wrap_into(q_c, *self.limits["alpha_axis_c"])
                if not self._in_limits("alpha_axis_c", q_c):
                    reason = "elbow outside axis_c limits"
                    continue
                inner = self.t_c + self.W * np.exp(1j * q_c)
                q_d = np.angle(inner) - np.angle(u)
                q_d = self._wrap_into(float(q_d), *self.limits["alpha_axis_d"])
                if not self._in_limits("alpha_axis_d", q_d):
                    reason = "shoulder outside axis_d limits"
                    continue

                q = {"alpha_axis_e": q_e, "alpha_axis_d": q_d, "alpha_axis_c": q_c}
                q["alpha_axis_b"] = (seed.get("alpha_axis_b", 0.0) if roll is None
                                     else self._wrap_into(float(roll),
                                                          *self.limits["alpha_axis_b"]))
                if not self._in_limits("alpha_axis_b", q["alpha_axis_b"]):
                    reason = "roll outside axis_b limits"
                    continue

                cost = sum((q[n] - seed[n]) ** 2 for n in q if n in seed) if seed else 0.0
                if cost < best_cost:
                    best, best_cost = q, cost

        if best is None:
            return None, {"reason": reason, "radial": radial}
        err = float(np.linalg.norm(self.fk(best)[0] - p))
        return best, {"reason": "ok", "error": err, "cost": best_cost}

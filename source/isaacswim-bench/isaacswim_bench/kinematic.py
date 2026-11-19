from __future__ import annotations

import math

import numpy as np

from isaacswim.np_math import quat_apply, quat_apply_inverse, wrap_pi

from .actions import Setpoints
from .core import SceneSpec, World, yaw_quat
from .envbase import BenchEnv
from .platform import (
    ARM_JOINTS,
    GRIP_CLOSED,
    GRIP_OPEN,
    JOINT_RATE,
    SimpleArm,
    grip_angle_for_gap,
    jaw_direction_yaw,
    tool_world,
)

__all__ = ["KinematicEnv"]

GRIP_RATE = 1.0


class KinematicEnv(BenchEnv):
    def __init__(self, task, *, arm=None, tau: float = 0.5, v_max: float = 0.4,
                 substeps: int = 5, **kw):
        super().__init__(task, arm=arm or SimpleArm(), **kw)
        self.tau, self.v_max, self.substeps = float(tau), float(v_max), int(substeps)

    def _reset_plant(self, spec: SceneSpec) -> World:
        self._props = {p.name: p for p in spec.props}
        self._objects = {n: (np.array(spec.pose(n)[0]), np.array(spec.pose(n)[1]))
                         for n in self._props}
        self._joint_of = {j.body: j for j in spec.joints}
        self._angles = {j.name: _yaw(self._objects[j.body][1]) for j in spec.joints}
        self._pos = np.array(spec.spawn_pos, dtype=np.float64)
        self._yaw = float(spec.spawn_yaw)
        self._vel_w = np.zeros(3)
        self._yaw_rate = 0.0
        self._q = {n: float((spec.arm or {}).get(n, 0.0)) for n in ARM_JOINTS}
        self._jaw = GRIP_OPEN
        self._grip = False
        self._held: str | None = None
        self._hold = None
        self._t = 0.0
        return self._snapshot()

    def _advance(self, sp: Setpoints, dt: float) -> World:
        h = dt / self.substeps
        a = 1.0 - math.exp(-h / self.tau)
        for _ in range(self.substeps):
            step = (sp.pos - self._pos) * a
            n = np.linalg.norm(step)
            if n > self.v_max * h:
                step *= self.v_max * h / n
            self._vel_w = step / h
            self._pos = self._pos + step
            dy = wrap_pi(sp.yaw - self._yaw) * a
            self._yaw_rate = dy / h
            self._yaw = wrap_pi(self._yaw + dy)
            for n_ in ARM_JOINTS:
                e = sp.arm[n_] - self._q[n_]
                self._q[n_] += max(-JOINT_RATE * h, min(JOINT_RATE * h, e))
            self._grip = sp.grip
            want = GRIP_CLOSED if sp.grip else GRIP_OPEN
            self._jaw += max(-GRIP_RATE * h, min(GRIP_RATE * h, want - self._jaw))
            self._update_grasp()
            self._t += h
        return self._snapshot()

    def _quat(self):
        return np.asarray(yaw_quat(self._yaw))

    def _tool(self):
        return tool_world(self._pos, self._quat(), self.arm, self._q)

    def _update_grasp(self):
        tool, R = self._tool()
        if self._held is None and self._grip:
            for name, prop in self._props.items():
                if not prop.graspable:
                    continue
                p, q = self._objects[name]
                top = p[2] + prop.half_extents[2]
                width = 2.0 * prop.half_extents[1]
                contact = grip_angle_for_gap(width)
                d = p - tool
                lateral = float(np.linalg.norm(d[:2]))
                if (lateral < 0.015 and top - 0.03 < tool[2] < top + 0.01
                        and self._jaw >= contact - 0.02):
                    self._held = name
                    jyaw = jaw_direction_yaw(self.arm, self._q, self._quat())
                    self._hold = (quat_apply_inverse(self._quat(), p - tool),
                                  _yaw(q), jyaw, contact)
                    break
        if self._held is not None:
            offset, yaw0, jyaw0, contact = self._hold
            self._jaw = min(self._jaw, contact + 0.01)
            if not self._grip and self._jaw < contact - 0.02:
                self._drop(self._held)
                self._held = None
                return
            jyaw = jaw_direction_yaw(self.arm, self._q, self._quat())
            new_yaw = yaw0 + wrap_pi(jyaw - jyaw0)
            if self._held in self._joint_of:
                j = self._joint_of[self._held]
                self._objects[self._held] = (np.array(j.anchor, dtype=np.float64),
                                             np.asarray(yaw_quat(new_yaw)))
                self._angles[j.name] = wrap_pi(new_yaw)
            else:
                self._objects[self._held] = (tool + quat_apply(self._quat(), offset),
                                             np.asarray(yaw_quat(new_yaw)))

    def _drop(self, name: str) -> None:
        if name in self._joint_of:
            return
        p, q = self._objects[name]
        he = self._props[name].half_extents
        floor = self.spec.seabed_z
        for other, prop in self._props.items():
            if other == name or prop.kind != "static" or prop.shape != "box":
                continue
            c, oq = self._objects[other]
            rel = quat_apply_inverse(oq, p - c)
            ohe = prop.half_extents
            if abs(rel[0]) < ohe[0] and abs(rel[1]) < ohe[1]:
                floor = max(floor, c[2] + ohe[2])
        self._objects[name] = (np.array([p[0], p[1], floor + he[2]]), q)

    def _snapshot(self) -> World:
        quat = self._quat()
        tool, R = self._tool()
        return World(
            t=self._t, pos=self._pos.copy(), quat=quat,
            lin_vel_b=quat_apply_inverse(quat, self._vel_w),
            ang_vel_b=np.array([0.0, 0.0, self._yaw_rate]),
            arm_q=dict(self._q), grip_q=float(self._jaw), grip_closed=bool(self._grip),
            tool_pos=tool, tool_R=R,
            objects={n: (p.copy(), q.copy()) for n, (p, q) in self._objects.items()},
            joints=dict(self._angles), seabed_z=self.spec.seabed_z,
            grasped=self._held)


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


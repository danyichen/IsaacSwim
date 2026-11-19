from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from isaacswim.isaaclab import _lab
from isaacswim.np_math import quat_apply

__all__ = ["LinkView", "ChaseView", "CameraRig", "ShowcaseVideo"]


@dataclass
class LinkView:
    link: str
    eye: tuple = (0.0, 0.0, 0.0)
    look: tuple = (1.0, 0.0, 0.0)
    focal: float = 14.0

    def pose(self, robot, env: int = 0):
        i = robot.body_names.index(self.link)
        p = _lab.data(robot).body_pos_w[env, i].detach().cpu().numpy().astype(np.float64)
        q = _lab.data(robot).body_quat_w[env, i].detach().cpu().numpy().astype(np.float64)
        eye = p + quat_apply(q, np.asarray(self.eye, dtype=np.float64))
        return eye, eye + 0.5 * quat_apply(q, np.asarray(self.look, dtype=np.float64))


@dataclass
class ChaseView:
    eye: tuple = (1.20, -1.55, 0.52)
    look: tuple = (0.26, 0.00, -0.26)
    focal: float = 18.0
    follow_yaw: bool = False

    def pose(self, robot, env: int = 0):
        p = _lab.data(robot).root_pos_w[env].detach().cpu().numpy().astype(np.float64)
        eye, look = np.asarray(self.eye, dtype=np.float64), np.asarray(self.look, dtype=np.float64)
        if self.follow_yaw:
            w, x, y, z = _lab.data(robot).root_quat_w[env].detach().cpu().numpy().astype(np.float64)
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            c, s = np.cos(yaw), np.sin(yaw)
            rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            eye, look = rz @ eye, rz @ look
        return p + eye, p + look


class CameraRig:
    def __init__(self, robot, views: Mapping[str, LinkView | ChaseView], *,
                 size=(128, 128), water=None, prim_root: str = "/World/cam",
                 clipping=(0.01, 60.0)):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        from isaacswim.isaaclab.camera import UnderwaterCamera

        self.robot = robot
        self.views = dict(views)
        self.cameras, self.underwater = {}, {}
        for name, view in self.views.items():
            h, w = size[name] if isinstance(size, Mapping) else size
            cam = Camera(CameraCfg(
                prim_path=f"{prim_root}_{name}", update_period=0.0,
                height=int(h), width=int(w),
                data_types=["rgb"] + (["distance_to_camera"] if water is not None else []),
                spawn=sim_utils.PinholeCameraCfg(focal_length=float(view.focal),
                                                 clipping_range=tuple(clipping))))
            self.cameras[name] = cam
            self.underwater[name] = (UnderwaterCamera(cam, water, range_key="distance_to_camera")
                                     if water is not None else None)

    @property
    def names(self) -> list[str]:
        return list(self.views)

    def set_water(self, water) -> None:
        for uw in self.underwater.values():
            if uw is not None:
                uw.set_water(water)

    def pose(self) -> None:
        import torch

        for name, view in self.views.items():
            eye, target = view.pose(self.robot)
            cam = self.cameras[name]
            dev = cam.device
            cam.set_world_poses_from_view(
                torch.tensor(np.array([eye]), dtype=torch.float32, device=dev),
                torch.tensor(np.array([target]), dtype=torch.float32, device=dev))

    def grab(self, render: Callable[[], None], *, dt: float = 0.0, pose: bool = True,
             settle: int = 0) -> dict[str, np.ndarray]:
        from isaacswim.isaaclab.recorder import grab_frames

        if pose:
            self.pose()
        names = self.names
        frames = grab_frames([self.cameras[n] for n in names], render,
                             underwater=[self.underwater[n] for n in names], dt=dt,
                             settle=settle)
        return dict(zip(names, frames, strict=True))


class ShowcaseVideo:
    def __init__(self, robot, *, close: LinkView | None, chase: ChaseView | None = None,
                 height: int = 540, water=None, thrusters=None, current=None,
                 seabed_z: float = -0.55, prim_root: str = "/World/video_cam",
                 arrows: dict | None = None, specks: dict | None = None):
        h = int(height) // 2 * 2
        views = {"chase": chase or ChaseView()}
        size = {"chase": (h, h * 16 // 9 // 2 * 2)}
        if close is not None:
            views["close"], size["close"] = close, (h, h * 4 // 3 // 2 * 2)
        self.rig = CameraRig(robot, views, size=size,
                             water=water, prim_root=prim_root)
        self.robot = robot
        self.thrusters = thrusters
        self.current = current
        self._seabed_z = float(seabed_z)
        self._arrow_kw = {"scale": "log", **(arrows or {})}
        self._speck_kw = dict(specks or {})
        self.arrows = None
        self.specks = None
        self._shown = {"arrows": None, "specks": None}

    def _drawings(self):
        if self.arrows is None and self.thrusters is not None:
            from isaacswim.isaaclab.markers import ThrusterVisuals
            self.arrows = ThrusterVisuals(self.thrusters, self.robot, **self._arrow_kw)
        if self.specks is None and self.current is not None:
            from isaacswim.isaaclab.markers import CurrentVisuals
            self.specks = CurrentVisuals(self.current, self.robot, seabed_z=self._seabed_z,
                                         **self._speck_kw)

    def reset(self, current=None) -> None:
        if current is not None:
            self.current = current
        if self.specks is not None:
            self.specks.reset(current)

    def hide_drawings(self, specks: bool = True, arrows: bool = True) -> None:
        self._set_visible(False, specks=specks, arrows=arrows)

    def _set_visible(self, on: bool, specks: bool = True, arrows: bool = True) -> None:
        for key, drawing, hold in (("arrows", self.arrows and self.arrows.arrows, arrows),
                                   ("specks", self.specks and self.specks.markers, specks)):
            if drawing is not None and hold and self._shown[key] != on:
                drawing.set_visibility(on)
                self._shown[key] = on

    def advance(self, thrust=None, dt: float = 1.0 / 30.0) -> None:
        self._drawings()
        if self.arrows is not None and thrust is not None:
            t = thrust.detach().cpu().numpy() if hasattr(thrust, "detach") else thrust
            self.arrows.update(np.asarray(t, dtype=np.float64), dt)
        if self.specks is not None:
            self.specks.update(dt)

    def grab(self, render: Callable[[], None], *, thrust=None, dt: float | None = None,
             render_dt: float = 0.0) -> np.ndarray:
        self._drawings()
        self._set_visible(True)
        if dt is not None:
            self.advance(thrust, dt)
        f = self.rig.grab(render, dt=render_dt)
        if "close" not in f:
            return f["chase"]
        return np.concatenate([f["chase"], f["close"]], axis=1)

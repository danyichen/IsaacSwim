from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from isaacswim.isaaclab.cameras import ShowcaseVideo  # noqa: E402


class _Markers:
    def __init__(self):
        self.visible = True
        self.drawn = 0
        self.toggles = 0

    def set_visibility(self, on):
        self.visible = bool(on)
        self.toggles += 1

    def draw(self):
        if self.visible:
            self.drawn += 1


class _Arrows:
    def __init__(self):
        self.arrows = _Markers()

    def update(self, thrust, dt):
        self.arrows.draw()


class _Specks:
    def __init__(self):
        self.markers = _Markers()

    def update(self, dt):
        self.markers.draw()


class _Rig:
    def __init__(self, video):
        self.video = video
        self.saw = []

    def grab(self, render, dt=0.0):
        self.saw.append((self.video.arrows.arrows.visible,
                         self.video.specks.markers.visible))
        img = np.zeros((2, 2, 3), np.uint8)
        return {"chase": img, "close": img}


def _video():
    v = ShowcaseVideo.__new__(ShowcaseVideo)
    v.arrows, v.specks = _Arrows(), _Specks()
    v.thrusters = v.current = object()
    v._shown = {"arrows": None, "specks": None}
    v.rig = _Rig(v)
    return v


def test_drawings_update_every_frame_despite_hiding():
    v = _video()
    for _ in range(5):
        v.hide_drawings()
        frame = v.grab(lambda: None, thrust=np.zeros(8), dt=0.05)
    assert v.arrows.arrows.drawn == 5 and v.specks.markers.drawn == 5
    assert v.rig.saw == [(True, True)] * 5
    assert frame.shape == (2, 4, 3)


def test_drawings_kept_in_every_view_are_never_toggled():
    v = _video()
    for _ in range(5):
        v.hide_drawings(specks=False, arrows=False)
        v.grab(lambda: None, thrust=np.zeros(8), dt=0.05)
    assert v.arrows.arrows.toggles <= 1 and v.specks.markers.toggles <= 1
    assert v.rig.saw == [(True, True)] * 5


def test_chase_view_follows_heading_but_never_rolls():
    import math
    from types import SimpleNamespace

    import torch

    from isaacswim.isaaclab.cameras import ChaseView

    def robot(yaw, roll=0.0):
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        q = [cy * cr, cy * sr, sy * sr, sy * cr]
        return SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, -1.0]]), root_quat_w=torch.tensor([q])))

    v = ChaseView(eye=(-1.0, 0.0, 0.5), look=(1.0, 0.0, 0.0), follow_yaw=True)
    eye, look = v.pose(robot(math.pi / 2))
    assert np.allclose(eye, [1.0, 1.0, -0.5], atol=1e-6)
    assert np.allclose(look, [1.0, 3.0, -1.0], atol=1e-6)
    eye_rolled, _ = v.pose(robot(math.pi / 2, roll=0.4))
    assert np.allclose(eye_rolled[2], -0.5, atol=1e-6)
    fixed, _ = ChaseView(eye=(-1.0, 0.0, 0.5), follow_yaw=False).pose(robot(math.pi / 2))
    assert np.allclose(fixed, [0.0, 2.0, -0.5])

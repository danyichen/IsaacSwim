from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaacswim.isaaclab.markers import CurrentVisuals, ThrusterVisuals


@pytest.fixture
def swirl():
    v = object.__new__(CurrentVisuals)
    v.modes = [(2.0 * np.pi / L, a / (2.0 * np.pi / L))
               for L, a in ((1.20, 0.25), (0.55, 0.12), (0.25, 0.06))]
    v.t = 3.1
    return v


@pytest.fixture
def pts():
    return np.random.default_rng(0).uniform(-1.3, 1.3, (4000, 3))


def test_swirl_is_divergence_free(swirl, pts):
    one = np.ones(len(pts))
    h = 1e-4
    div = np.zeros(len(pts))
    for ax in (0, 1):
        d = np.zeros(3)
        d[ax] = h
        div += (swirl._swirl(pts + d, one)[:, ax]
                - swirl._swirl(pts - d, one)[:, ax]) / (2 * h)
    speed = np.linalg.norm(swirl._swirl(pts, one)[:, :2], axis=1)
    assert np.abs(div).max() < 1e-3 * max(speed.max(), 1e-9)


def test_swirl_actually_stirs(swirl, pts):
    near = pts[:200]
    spread = np.linalg.norm(swirl._swirl(near, np.ones(200))
                            - swirl._swirl(near + 0.05, np.ones(200)), axis=1)
    assert spread.mean() > 0.01


class _Markers:
    def __init__(self):
        self.drawn = None

    def visualize(self, translations, scales):
        self.drawn = translations.numpy().copy()


def _visuals(flow, root, *, n=300, box=(2.6, 2.6, 1.15)):
    cv = CurrentVisuals.__new__(CurrentVisuals)
    cv.gusts = lambda pos: torch.as_tensor(flow, dtype=torch.float32).expand_as(pos).clone()
    cv.robot = SimpleNamespace(device="cpu", data=SimpleNamespace(
        root_pos_w=torch.tensor([root], dtype=torch.float32)))
    cv.seabed_z = -2.0
    cv.box = np.asarray(box, dtype=float)
    cv.n = n
    cv._rng = np.random.default_rng(0)
    cv.p = None
    cv.modes = []
    cv._src = []
    cv.emit = 0.0
    cv.jitter = 0.0
    cv.jitter_tau = 0.9
    cv.j = np.zeros((n, 3))
    cv.t = 0.0
    cv.scales = np.ones((n, 3))
    cv.markers = _Markers()
    return cv


def _move(cv, root):
    cv.robot.data.root_pos_w = torch.tensor([root], dtype=torch.float32)


def test_still_water_specks_stay_put_when_the_vehicle_moves():
    cv = _visuals((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    cv.update(0.05)
    before = cv.markers.drawn
    _move(cv, (0.30, -0.20, -1.0))
    cv.update(0.05)
    after = cv.markers.drawn
    stayed = np.all(np.isclose(before, after, atol=1e-5), axis=1)
    assert stayed.mean() > 0.7
    jumped = np.linalg.norm((after - before)[~stayed], axis=1)
    assert np.all(jumped > 1.0)


def test_specks_drift_with_the_current_in_world():
    flow = (0.2, 0.0, 0.0)
    cv = _visuals(flow, (0.0, 0.0, -1.0))
    cv.update(0.1)
    before = cv.markers.drawn
    _move(cv, (0.0, 0.5, -1.0))
    cv.update(0.1)
    d = cv.markers.drawn - before
    kept = np.linalg.norm(d, axis=1) < 0.5
    assert np.allclose(d[kept], [0.02, 0.0, 0.0], atol=1e-5)


def test_box_follows_depth_but_not_below_the_seabed():
    cv = _visuals((0.0, 0.0, 0.0), (0.0, 0.0, 3.0))
    cv.update(0.05)
    z = cv.markers.drawn[:, 2]
    assert z.min() >= 3.0 - 0.575 - 1e-6 and z.max() <= 3.0 + 0.575 + 1e-6
    _move(cv, (0.0, 0.0, -1.9))
    cv.update(0.05)
    assert cv.markers.drawn[:, 2].min() >= -2.0 + 0.02 - 1e-6


def _arrows(scale, ref_thrust=None):
    v = ThrusterVisuals.__new__(ThrusterVisuals)
    v.scale, v.knee = scale, 1.0
    v.ref_thrust = ref_thrust if ref_thrust is not None else (51.5 if scale == "log" else 2.5)
    return v


def test_linear_arrows_pin_above_the_reference():
    v = _arrows("linear")
    assert [min(1.0, v._fraction(t)) for t in (5.0, 7.0, 23.0)] == [1.0, 1.0, 1.0]


def test_log_arrows_still_move_across_the_working_range():
    v = _arrows("log")
    f = [v._fraction(t) for t in (0.0, 0.5, 2.5, 7.0, 23.0, 51.5)]
    assert f[0] == 0.0 and f[-1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(f[:-1], f[1:], strict=True))
    assert f[1:5] == pytest.approx([0.10, 0.32, 0.52, 0.80], abs=0.01)

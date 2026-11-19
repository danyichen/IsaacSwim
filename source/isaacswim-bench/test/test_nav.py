from __future__ import annotations

import math

import numpy as np
import pytest

from isaacswim.np_math import quat_apply, rpy_to_quat
from isaacswim_bench.core import Command
from isaacswim_bench.envbase import BenchEnv
from isaacswim_bench.nav import DeadReckoning, Instruments, NavConfig, yaw_rotate

Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def _fly(nav, vel_b, quat, seconds, dt=1 / 120, dvl_every=12, z=0.0, valid=True):
    for k in range(int(round(seconds / dt))):
        nav.update(dt, quat=quat, ang_vel_b=np.zeros(3), depth_z=z,
                   dvl=vel_b if k % dvl_every == 0 else None, dvl_valid=valid)


def test_perfect_instruments_dead_reckon_exactly():
    nav = DeadReckoning(NavConfig())
    nav.reset([0.0, 0.0, 0.0], Q0)
    q = np.asarray(rpy_to_quat((0.0, 0.0, math.pi / 2)))
    _fly(nav, np.array([0.2, 0.0, 0.0]), q, 5.0)
    assert nav.pos[:2] == pytest.approx([0.0, 1.0], abs=1e-9)


def test_drift_is_scale_and_heading_error_times_distance():
    nav = DeadReckoning(NavConfig())
    nav.reset(np.zeros(3), Q0)
    b = math.radians(1.0)
    _fly(nav, np.array([0.2 * 1.01, 0.0, 0.0]), yaw_rotate(Q0, b), 10.0)
    assert nav.pos[0] == pytest.approx(2.02 * math.cos(b), abs=1e-6)
    assert nav.pos[1] == pytest.approx(2.02 * math.sin(b), abs=1e-6)


def test_depth_follows_the_barometer_on_average_and_the_dvl_in_between():
    nav = DeadReckoning(NavConfig(depth_tau=0.5))
    nav.reset(np.array([0.0, 0.0, -1.0]), Q0)
    rng = np.random.default_rng(0)
    zs = []
    for _ in range(600):
        nav.update(1 / 120, quat=Q0, ang_vel_b=np.zeros(3),
                   depth_z=-1.0 + rng.normal(0, 0.01), dvl=np.zeros(3))
        zs.append(nav.pos[2])
    assert np.mean(zs) == pytest.approx(-1.0, abs=2e-3)
    assert np.std(zs) < 0.01 / 5


def test_out_of_lock_the_velocity_decays_instead_of_running_away():
    nav = DeadReckoning(NavConfig(dropout_tau=1.0))
    nav.reset(np.zeros(3), Q0)
    _fly(nav, np.array([0.3, 0.0, 0.0]), Q0, 1.0)
    x = nav.pos[0]
    _fly(nav, np.zeros(3), Q0, 10.0, valid=False)
    assert not nav.lock
    assert nav.pos[0] - x == pytest.approx(0.3 * 1.0, rel=0.05)
    assert np.abs(nav.vel_b).max() < 1e-4


def test_instruments_draw_per_dive_errors_from_their_own_stream():
    a = Instruments(NavConfig(), np.random.default_rng(1))
    b = Instruments(NavConfig(), np.random.default_rng(1))
    assert a.heading_bias == b.heading_bias and a.dvl_scale == b.dvl_scale
    assert a.heading_bias != 0.0 and a.dvl_scale != 1.0
    off = Instruments(NavConfig().scaled(0.0), np.random.default_rng(1))
    assert off.heading_bias == 0.0 and off.dvl_scale == 1.0


def test_no_bottom_lock_above_range():
    ins = Instruments(NavConfig(), np.random.default_rng(0), max_altitude=30.0)
    v, ok = ins.dvl([0.1, 0.0, 0.0], altitude=40.0)
    assert not ok and not v.any()


class _Env(BenchEnv):
    def __init__(self, world_pos, nav_pos, bias):
        self.world = type("W", (), {"pos": np.asarray(world_pos, float), "yaw": 0.0})()
        self._np, self._b = np.asarray(nav_pos, float), bias

    def _nav_pose(self):
        return self._np, self._b

    def _nav_heading_error(self):
        return self._b


def test_expert_commands_land_where_meant_when_flown_on_biased_navigation():
    b = math.radians(3.0)
    env = _Env([1.0, 2.0, -0.5], [1.05, 1.98, -0.5], b)
    want_w = np.array([1.6, 2.3, -0.4])
    cmd = env.to_nav(Command(pos=want_w, yaw=0.2))
    assert cmd.yaw == pytest.approx(0.2 + b)
    err_n = cmd.pos - env._np
    moved_w = quat_apply(np.asarray(rpy_to_quat((0, 0, -b))), err_n)
    assert env.world.pos + moved_w == pytest.approx(want_w, abs=1e-12)

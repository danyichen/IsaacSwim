from __future__ import annotations

import math

import torch

from isaacswim import FluidShape, Medium
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, _lab

E, B = 2, 1
Q_WXYZ = torch.tensor([math.cos(math.pi / 12), 0.0, 0.0, math.sin(math.pi / 12)])


class Proxy:
    def __init__(self, t):
        self.torch = t


class Data2:
    def __init__(self):
        self.body_link_quat_w = Q_WXYZ.expand(E, B, 4).clone()
        self.body_link_pos_w = torch.zeros(E, B, 3)
        self.body_link_lin_vel_w = torch.tensor([0.4, 0.1, 0.0]).expand(E, B, 3).clone()
        self.body_link_ang_vel_w = torch.tensor([0.0, 0.0, 0.3]).expand(E, B, 3).clone()
        self.root_pos_w = self.body_link_pos_w[:, 0]
        self.default_root_state = torch.cat([torch.zeros(E, 3), Q_WXYZ.expand(E, 4),
                                             torch.zeros(E, 6)], -1)


class Data3:
    def __init__(self):
        d2 = Data2()
        xyzw = _lab.quat_to_lab(Q_WXYZ)
        self.body_link_quat_w = Proxy(xyzw.expand(E, B, 4).clone())
        self.body_link_pos_w = Proxy(d2.body_link_pos_w)
        self.body_link_lin_vel_w = Proxy(d2.body_link_lin_vel_w)
        self.body_link_ang_vel_w = Proxy(d2.body_link_ang_vel_w)
        self.root_link_pos_w = Proxy(d2.root_pos_w)
        self.default_root_state = Proxy(torch.cat([torch.zeros(E, 3), xyzw.expand(E, 4),
                                                   torch.zeros(E, 6)], -1))


class Composer3:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques_index(self, forces=None, torques=None, is_global=False, **kw):
        self.calls.append((forces.clone(), torques.clone(), is_global))


class Asset:
    def __init__(self, lab3: bool):
        self.body_names = ["hull"]
        self.device = "cpu"
        self.data = Data3() if lab3 else Data2()
        self.writes = []
        if lab3:
            self.permanent_wrench_composer = Composer3()

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
        self.writes.append(("pose3", root_pose.clone(), env_ids))

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.writes.append(("pose2", pose.clone(), env_ids))

    def set_external_force_and_torque(self, F, T, is_global=False):
        self.writes.append(("wrench2", F.clone(), T.clone()))


def test_detection_and_reads_are_wxyz_on_both():
    a2, a3 = Asset(False), Asset(True)
    assert not _lab.is_lab3(a2) and _lab.is_lab3(a3)
    for a in (a2, a3):
        d = _lab.data(a)
        assert torch.allclose(d.body_link_quat_w[0, 0], Q_WXYZ)
        assert torch.allclose(d.default_root_state[0, 3:7], Q_WXYZ)
        assert isinstance(d.body_link_lin_vel_w, torch.Tensor)


def test_pose_write_converts_for_lab3_only():
    pose = torch.cat([torch.tensor([[1.0, 2.0, 3.0]]), Q_WXYZ[None]], -1)
    a2, a3 = Asset(False), Asset(True)
    _lab.write_root_pose(a2, pose, env_ids=[0])
    _lab.write_root_pose(a3, pose, env_ids=[0])
    assert a2.writes[0][0] == "pose2" and torch.allclose(a2.writes[0][1], pose)
    assert a3.writes[0][0] == "pose3"
    assert torch.allclose(a3.writes[0][1][0, 3:], _lab.quat_to_lab(Q_WXYZ))


def test_hydrodynamics_wrench_identical_on_both():
    cfg = dict(shapes=[FluidShape(body="hull", semiaxes=(0.3, 0.2, 0.15))],
               medium=Medium(density=1000.0, gravity=(0.0, 0.0, 0.0)))
    F2, T2 = Hydrodynamics(HydrodynamicsCfg(**cfg), Asset(False)).apply()
    a3 = Asset(True)
    F3, T3 = Hydrodynamics(HydrodynamicsCfg(**cfg), a3).apply()
    assert torch.allclose(F2, F3, atol=1e-6) and torch.allclose(T2, T3, atol=1e-6)
    assert float(F3.abs().sum()) > 0.0
    assert len(a3.permanent_wrench_composer.calls) == 1


def test_init_rot_order():
    assert _lab.init_rot((1, 0, 0, 0), lab3=True) == (0.0, 0.0, 0.0, 1.0)
    assert _lab.init_rot((1, 0, 0, 0), lab3=False) == (1.0, 0.0, 0.0, 0.0)

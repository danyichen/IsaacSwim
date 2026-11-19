from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def vector_class(monkeypatch):
    parent = "isaacswim_bench.sim"
    module = types.ModuleType(parent)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, parent, module)
    monkeypatch.setitem(sys.modules, parent + ".env", types.SimpleNamespace(SimEnv=object))
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", types.ModuleType("isaaclab.sim"))
    path = Path(__file__).parents[1] / "isaacswim_bench/sim/vector.py"
    spec = importlib.util.spec_from_file_location(parent + ".vector", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.VectorSimEnv


class Slot:
    substeps = 3
    physics_dt = 0.1
    control_dt = 0.3

    def __init__(self, index, shared):
        self.index, self.shared = index, shared
        self.position = 0.0
        self.velocity = 0.0
        self._t = 0.0
        self.world = types.SimpleNamespace(t=0.0)
        self.actions = types.SimpleNamespace(
            apply=lambda action, **_: types.SimpleNamespace(pos=action[0], yaw=0, wrench=None)
        )

    def reset(self, seed):
        self.position = float(seed)
        self.velocity = self._t = 0.0
        self.world = self._snapshot()
        return {"position": np.array([self.position])}, {"seed": seed}

    def _nav_pose(self):
        return [self.position, 0, 0], 0

    def split_action(self, action):
        return action, None

    def _begin_control(self, command, dt):
        return np.zeros(3), 0

    def _before_physics(self, position, *_args, **_kwargs):
        assert self.index not in self.shared.forces
        self.shared.forces[self.index] = position

    def _after_physics(self):
        assert not self.shared.forces

    def _snapshot(self):
        return types.SimpleNamespace(t=self._t, position=self.position)

    def _finish_step(self, command):
        return {"position": np.array([self.position])}, self.position, False, False, {}


class Clock:
    def __init__(self):
        self.forces = {}
        self.ticks = 0
        self.slots = []

    def step(self, render):
        assert len(self.forces) == len(self.slots)
        self.ticks += 1
        for slot in self.slots:
            slot.velocity += self.forces[slot.index] * slot.physics_dt
            slot.position += slot.velocity * slot.physics_dt
        self.forces.clear()


def construct(cls):
    vector = cls.__new__(cls)
    vector.sim = Clock()
    vector.num_envs = 2
    vector.action_size = 1
    vector.envs = [Slot(i, vector.sim) for i in range(2)]
    vector.sim.slots = vector.envs
    vector._ready = np.ones(2, dtype=bool)
    vector._done = np.zeros(2, dtype=bool)
    vector.observations = [{"position": np.array([0.0])} for _ in range(2)]
    return vector


def test_all_slot_forces_precede_each_single_shared_step(vector_class):
    env = construct(vector_class)
    observations, rewards, done, truncated, info = env.step([[1.0], [-2.0]])
    assert env.sim.ticks == 3
    np.testing.assert_allclose(observations["position"], [[0.06], [-0.12]])
    np.testing.assert_allclose(rewards, [0.06, -0.12])
    np.testing.assert_allclose([slot.world.t for slot in env.envs], [0.3, 0.3])
    assert not done.any() and not truncated.any()


def test_selective_reset_preserves_peer_motion_and_clock(vector_class):
    env = construct(vector_class)
    env.step([[1.0], [-2.0]])
    peer = (env.envs[1].position, env.envs[1].velocity, env.envs[1].world.t)
    observations, info = env.reset(seed=[8], env_ids=[0])
    assert env.sim.ticks == 3
    assert (env.envs[1].position, env.envs[1].velocity, env.envs[1].world.t) == peer
    np.testing.assert_array_equal(observations["position"], [[8.0]])
    assert info == [{"seed": 8}]


def test_empty_reset_is_a_noop_and_numpy_seed_is_accepted(vector_class):
    env = construct(vector_class)
    assert env.reset(env_ids=[]) == ({}, [])
    assert env.sim.ticks == 0
    observations, info = env.reset(seed=np.int64(12), env_ids=[1])
    np.testing.assert_array_equal(observations["position"], [[13.0]])
    assert info == [{"seed": 13}]


def test_initial_settling_defines_the_episode_start_state(vector_class, monkeypatch):
    env = construct(vector_class)
    env._ready[:] = False
    monkeypatch.setitem(vector_class.reset.__globals__, "_Stats", lambda world: world.position)
    for slot in env.envs:
        slot._settle_target = (1.0, 0.0)
        slot.vehicle = types.SimpleNamespace(time=0.0)
        slot.params = {}
        slot.actions.reset = lambda *args, **kwargs: None
        slot.world.arm_q = {}
        slot.task = types.SimpleNamespace(reset=lambda params, world: world.position)
        slot.observe = lambda slot=slot: {"position": np.array([slot.position])}
        snapshot = slot._snapshot
        slot._snapshot = lambda snapshot=snapshot: types.SimpleNamespace(
            **vars(snapshot()), arm_q={}
        )
    observations, _ = env.reset(seed=0)
    assert env.sim.ticks == 5
    np.testing.assert_allclose(observations["position"], [[0.15], [1.15]])
    np.testing.assert_allclose([slot._mem for slot in env.envs], [0.15, 1.15])
    assert all(slot.world.t == slot.vehicle.time == 0.0 for slot in env.envs)

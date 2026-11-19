from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np

from .. import make_task
from ..envbase import _Stats
from .env import SimEnv


def stack_observations(observations):
    result = {}
    keys = observations[0].keys()
    for key in keys:
        values = [obs[key] for obs in observations]
        if all(isinstance(v, np.ndarray) and v.shape == values[0].shape for v in values):
            result[key] = np.stack(values)
        else:
            result[key] = tuple(values)
    return result


class VectorSimEnv:
    def __init__(
        self, task, num_envs=1, *, device="cuda:0", spacing=250.0, physics_dt=1.0 / 120.0, **kwargs
    ):
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(spacing) or spacing < 50.0:
            raise ValueError("spacing must be finite and at least 50 m")
        names = [task] * num_envs if isinstance(task, str) else list(task)
        if len(names) != num_envs:
            raise ValueError("one task name is required per environment")
        tasks = [make_task(name) for name in names]
        seabed = tasks[0].seabed_z
        if any(t.seabed_z != seabed for t in tasks):
            raise ValueError("shared environments require the same seabed height")
        self.num_envs = num_envs
        render = {}
        if hasattr(sim_utils, "RenderCfg"):
            antialiasing = kwargs.get("antialiasing", "auto")
            render["render"] = sim_utils.RenderCfg(
                antialiasing_mode="FXAA" if antialiasing == "auto" else antialiasing
            )
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=physics_dt,
                device=device,
                gravity=(0.0, 0.0, -9.81),
                enable_scene_query_support=bool(kwargs.get("oceansim")),
                **render,
            )
        )
        light = sim_utils.DomeLightCfg(intensity=900.0, color=(0.62, 0.78, 0.92))
        light.func("/World/DomeLight", light)
        ground = sim_utils.GroundPlaneCfg(color=(0.28, 0.26, 0.22), size=(10000, 10000))
        ground.func("/World/seabed", ground, translation=(0.0, 0.0, seabed))
        width = math.ceil(math.sqrt(num_envs))
        self.envs = [
            SimEnv(
                t,
                sim=self.sim,
                device=device,
                physics_dt=physics_dt,
                root=f"/World/envs/env_{i}",
                origin=((i % width) * spacing, (i // width) * spacing, 0.0),
                defer_initialize=True,
                **kwargs,
            )
            for i, t in enumerate(tasks)
        ]
        self.sim.reset()
        for env in self.envs:
            env.initialize()
        self.device = self.sim.device
        self.action_size = self.envs[0].action_size
        self.control_dt = self.envs[0].control_dt
        self.observations = [None] * num_envs
        self._ready = np.zeros(num_envs, dtype=bool)
        self._done = np.zeros(num_envs, dtype=bool)

    def reset(self, *, seed=0, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("env_ids must contain unique valid slot indices")
        if not ids:
            return {}, []
        seeds = [int(seed) + i for i in ids] if isinstance(seed, (int, np.integer)) else list(seed)
        if len(seeds) != len(ids):
            raise ValueError("one seed is required per reset slot")
        infos = []
        first_reset = not self._ready.any()
        for i, value in zip(ids, seeds):
            observation, info = self.envs[i].reset(seed=int(value))
            self.observations[i] = observation
            infos.append(info)
            self._ready[i] = True
            self._done[i] = False
        if first_reset and self._ready.all():
            for _ in range(round(0.5 / self.envs[0].physics_dt)):
                for env in self.envs:
                    env._before_physics(*env._settle_target, None)
                self.sim.step(render=False)
                for env in self.envs:
                    env._after_physics()
            for env in self.envs:
                env._t = env.vehicle.time = 0.0
                env.world = env._snapshot()
                env.actions.reset(*env._nav_pose(), env.world.arm_q, grip=False)
                env._mem = env.task.reset(env.params, env.world)
                env.stats = _Stats(env.world)
            self.observations = [env.observe() for env in self.envs]
        return stack_observations([self.observations[i] for i in ids]), infos

    def expert_action(self):
        return np.stack([env.expert_action() for env in self.envs])

    def step(self, actions):
        if not self._ready.all():
            raise RuntimeError("reset every slot before stepping")
        if self._done.any():
            raise RuntimeError("reset completed slots before stepping again")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_size) or not np.isfinite(actions).all():
            raise ValueError(
                f"actions must be finite with shape {(self.num_envs, self.action_size)}"
            )
        commands, motion, readings = [], [], []
        for env, action in zip(self.envs, actions):
            control, reading = env.split_action(action)
            readings.append(reading)
            pos, yaw = env._nav_pose()
            command = env.actions.apply(control, pos=pos, yaw=yaw)
            commands.append(command)
            motion.append(env._begin_control(command, env.control_dt))
        for k in range(self.envs[0].substeps):
            for env, command, (velocity, rate) in zip(self.envs, commands, motion):
                env._before_physics(
                    command.pos, command.yaw, command.wrench, velocity, rate, props_all=k == 0
                )
            self.sim.step(render=False)
            for env in self.envs:
                env._after_physics()
        results = []
        for env, command, reading in zip(self.envs, commands, readings):
            env._t += env.substeps * env.physics_dt
            env.world = env._snapshot()
            env.world.meter_reading = reading
            results.append(env._finish_step(command))
        obs, rewards, terminated, truncated, infos = zip(*results)
        self.observations = list(obs)
        self._done = np.asarray(terminated) | np.asarray(truncated)
        return (
            stack_observations(obs),
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            list(infos),
        )

    def reports(self):
        return [env.report() for env in self.envs]

    def close(self):
        if getattr(self, "_closed", False):
            return
        self.sim.pause()
        if hasattr(self.sim, "clear_all_callbacks"):
            self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        self._closed = True

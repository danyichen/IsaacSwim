from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaacswim_bench.sim import VectorSimEnv


@configclass
class TaskEnvCfg(DirectRLEnvCfg):
    task_name = "sample_collect"
    episode_length_s = 180.0
    decimation = 6
    action_space = 9
    observation_space = 1
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=250.0)
    views: tuple = ()


@configclass
class StationTaskEnvCfg(TaskEnvCfg):
    task_name = "station_keep"


@configclass
class PipelineEnvCfg(TaskEnvCfg):
    task_name = "pipeline_inspect"


@configclass
class TargetEnvCfg(TaskEnvCfg):
    task_name = "target_approach"


@configclass
class DockEnvCfg(TaskEnvCfg):
    task_name = "dock"


@configclass
class SampleEnvCfg(TaskEnvCfg):
    task_name = "sample_collect"


@configclass
class ValveEnvCfg(TaskEnvCfg):
    task_name = "valve_turn"


class TaskEnv(DirectRLEnv):
    def __init__(self, cfg: TaskEnvCfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.render_mode = render_mode
        self._is_closed = False
        self.vector = VectorSimEnv(
            cfg.task_name,
            cfg.scene.num_envs,
            device=cfg.sim.device,
            spacing=cfg.scene.env_spacing,
            physics_dt=cfg.sim.dt,
            control_hz=1 / (cfg.sim.dt * cfg.decimation),
            views=cfg.views,
        )
        self.sim = self.vector.sim
        self.scene = SimpleNamespace(num_envs=cfg.scene.num_envs)
        self._seed = int(cfg.seed or 0)
        self._next_seed = self._seed
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.cfg.episode_length_s = self.vector.envs[0].task.horizon
        self._obs, _ = self.vector.reset(seed=self._seed)
        self._next_seed += self.num_envs
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.vector.action_size,), dtype=np.float32
        )
        size = self._get_observations()["policy"].shape[-1]
        self.cfg.observation_space = size
        self.cfg.action_space = self.vector.action_size
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)}
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.extras = {}

    @property
    def num_envs(self):
        return self.vector.num_envs

    @property
    def device(self):
        return self.vector.device

    @property
    def step_dt(self):
        return self.vector.control_dt

    @property
    def max_episode_length_s(self):
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        return math.ceil(self.max_episode_length_s / self.step_dt)

    def seed(self, seed=-1):
        self._seed = int(seed) if seed >= 0 else self._seed
        self._next_seed = self._seed
        return self._seed

    @property
    def obs_buf(self):
        return self._get_observations()

    def _get_observations(self):
        state = np.concatenate(
            [self._obs[k] for k in ("proprio", "priv_odom", "priv_task")], axis=-1
        )
        return {"policy": torch.as_tensor(state, device=self.device, dtype=torch.float32)}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._obs, _ = self.vector.reset(seed=self._next_seed)
        self._next_seed += self.num_envs
        self.episode_length_buf.zero_()
        return self._get_observations(), {}

    def step(self, actions):
        self._obs, rewards, terminated, truncated, _ = self.vector.step(
            actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        )
        self.episode_length_buf += 1
        expired = (self.episode_length_buf >= self.max_episode_length).cpu().numpy()
        truncated |= expired & ~terminated
        reset = np.flatnonzero(terminated | truncated).tolist()
        reports = [self.vector.envs[i].report() for i in reset]
        if reset:
            seeds = list(range(self._next_seed, self._next_seed + len(reset)))
            self._next_seed += len(reset)
            changed, _ = self.vector.reset(seed=seeds, env_ids=reset)
            for key in self._obs:
                if isinstance(self._obs[key], np.ndarray):
                    self._obs[key][reset] = changed[key]
                else:
                    values = list(self._obs[key])
                    for j, i in enumerate(reset):
                        values[i] = changed[key][j]
                    self._obs[key] = tuple(values)
            self.episode_length_buf[reset] = 0
        self.extras = {
            "log": {"Episode/success": float(np.mean([r["success"] for r in reports]))}
            if reports
            else {}
        }
        return (
            self._get_observations(),
            torch.as_tensor(rewards, device=self.device).float(),
            torch.as_tensor(terminated, device=self.device),
            torch.as_tensor(truncated, device=self.device),
            self.extras,
        )

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self.vector.close()

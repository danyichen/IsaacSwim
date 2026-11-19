from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaacswim.actuators import Motor, Propeller, ThrusterArray, Thrusters
from isaacswim.assets.bluerov import ARM_HOME, JAWS_USD, build_hydro
from isaacswim.isaaclab import Vehicle, _lab
from isaacswim.torch_math import quat_apply_inverse

__all__ = ["StationKeepEnv", "StationKeepEnvCfg", "StationKeepSceneCfg"]


@configclass
class StationKeepSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(JAWS_USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=ARM_HOME),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0,
                                              damping=30.0)})


@configclass
class StationKeepEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_space = 8
    observation_space = 22
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4,
                                       gravity=(0.0, 0.0, -9.81))
    scene: StationKeepSceneCfg = StationKeepSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True)

    spawn_radius = 1.5
    max_current = 0.3
    esc_time_constant = 0.05
    voltage = 16.0
    terminate_distance = 4.0

    w_pos = 1.0
    w_yaw = 0.5
    w_lin_vel = -0.05
    w_ang_vel = -0.02
    w_thrust = -0.01
    w_rate = -0.02


class StationKeepEnv(DirectRLEnv):
    cfg: StationKeepEnvCfg

    def __init__(self, cfg: StationKeepEnvCfg, render_mode: str | None = None, **kwargs):
        usd = Path(cfg.scene.robot.spawn.usd_path)
        if not usd.is_file():
            raise FileNotFoundError(
                f"{usd} not found -- build the vehicle first: "
                "isaaclab.sh -p scripts/tools/build_bluerov.py")
        super().__init__(cfg, render_mode, **kwargs)
        E, dev = self.num_envs, self.device
        self._current = torch.zeros(E, 3, device=dev)
        hydro, _ = build_hydro(self._robot, usd.with_suffix(".urdf"), surface_z=None,
                               seabed_z=None, device=dev)
        hydro.cfg.current_fn = self._current_at
        hydro.set_medium(hydro.cfg.medium)
        prop = Propeller.t200(self.cfg.voltage)
        self._thrusters = Thrusters(ThrusterArray(usd.parent / "thrusters.json"),
                                    body="base_link", input="thrust", propeller=prop,
                                    motor=Motor.t200(self.cfg.voltage),
                                    time_constant=self.cfg.esc_time_constant)
        self.vehicle = Vehicle(self._robot, hydro=hydro, actuators=[self._thrusters])
        arr = self._thrusters.array
        self._scale_fwd, self._scale_rev = arr.fwd, arr.rev
        self._actions = torch.zeros(E, 8, device=dev)
        self._prev = torch.zeros(E, 8, device=dev)
        self._target_pos = torch.zeros(E, 3, device=dev)
        self._target_yaw = torch.zeros(E, device=dev)
        self._sums = {k: torch.zeros(E, device=dev)
                      for k in ("pos", "yaw", "lin_vel", "ang_vel", "thrust", "rate")}

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        light = sim_utils.DomeLightCfg(intensity=900.0, color=(0.62, 0.78, 0.92))
        light.func("/World/Light", light)

    def _current_at(self, pos: torch.Tensor) -> torch.Tensor:
        c = self._current.to(pos.dtype)
        return c.view(c.shape[0], *([1] * (pos.dim() - 2)), 3).expand_as(pos).clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)
        a = self._actions.double()
        thrust = torch.where(a >= 0, a * self._scale_fwd, a * self._scale_rev)
        self._thrusters.set_setpoint(thrust)

    def _apply_action(self):
        self.vehicle.apply(self.physics_dt)

    def _errors(self):
        d = _lab.data(self._robot)
        pos_b = quat_apply_inverse(d.root_quat_w, self._target_pos - d.root_pos_w)
        q = d.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        yaw_err = torch.atan2(torch.sin(self._target_yaw - yaw),
                              torch.cos(self._target_yaw - yaw))
        return pos_b, yaw_err

    def _get_observations(self) -> dict:
        d = _lab.data(self._robot)
        pos_b, yaw_err = self._errors()
        g_b = quat_apply_inverse(d.root_quat_w, torch.tensor(
            [[0.0, 0.0, -1.0]], device=self.device).expand(self.num_envs, 3))
        obs = torch.cat([pos_b, g_b, torch.sin(yaw_err)[:, None], torch.cos(yaw_err)[:, None],
                         d.root_lin_vel_b, d.root_ang_vel_b, self._actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = _lab.data(self._robot)
        pos_b, yaw_err = self._step_errors
        dist = pos_b.norm(dim=-1)
        terms = {
            "pos": self.cfg.w_pos * (1.0 - torch.tanh(dist / 0.5)),
            "yaw": self.cfg.w_yaw * (1.0 - torch.tanh(yaw_err.abs() / 0.5)),
            "lin_vel": self.cfg.w_lin_vel * d.root_lin_vel_b.square().sum(-1),
            "ang_vel": self.cfg.w_ang_vel * d.root_ang_vel_b.square().sum(-1),
            "thrust": self.cfg.w_thrust * self._actions.square().sum(-1),
            "rate": self.cfg.w_rate * (self._actions - self._prev).square().sum(-1),
        }
        for k, v in terms.items():
            self._sums[k] += v * self.step_dt
        return torch.stack(list(terms.values())).sum(0) * self.step_dt

    def _get_dones(self):
        self._step_errors = self._errors()
        pos_b, _ = self._step_errors
        far = pos_b.norm(dim=-1) > self.cfg.terminate_distance
        return far, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids = _lab.to_torch(env_ids).to(self.device, torch.long)
        pos_b, yaw_err = getattr(self, "_step_errors", None) or self._errors()
        log = {f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s
               for k, v in self._sums.items()}
        log["Metrics/final_distance"] = pos_b[env_ids].norm(dim=-1).mean()
        log["Metrics/final_yaw_error"] = yaw_err[env_ids].abs().mean()
        log["Episode_Termination/far"] = torch.count_nonzero(self.reset_terminated[env_ids])
        self.extras["log"] = log
        self._step_errors = None
        for v in self._sums.values():
            v[env_ids] = 0.0

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length))
        n = len(env_ids)
        origin = self.scene.env_origins[env_ids]
        self._target_pos[env_ids] = origin
        self._target_yaw[env_ids] = (torch.rand(n, device=self.device) * 2 - 1) * math.pi
        offset = torch.randn(n, 3, device=self.device)
        offset *= (torch.rand(n, 1, device=self.device) * self.cfg.spawn_radius
                   / offset.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        yaw0 = (torch.rand(n, device=self.device) * 2 - 1) * math.pi
        root = _lab.data(self._robot).default_root_state[env_ids].clone()
        root[:, :3] = origin + offset
        root[:, 3:7] = torch.stack([torch.cos(yaw0 / 2), torch.zeros_like(yaw0),
                                    torch.zeros_like(yaw0), torch.sin(yaw0 / 2)], -1)
        root[:, 7:] = 0.0
        _lab.write_root_pose(self._robot, root[:, :7], env_ids)
        _lab.write_root_velocity(self._robot, root[:, 7:], env_ids)
        d = _lab.data(self._robot)
        _lab.write_joint_state(self._robot, d.default_joint_pos[env_ids],
                               d.default_joint_vel[env_ids], env_ids)
        ang = torch.rand(n, device=self.device) * 2 * math.pi
        mag = torch.rand(n, device=self.device) * self.cfg.max_current
        self._current[env_ids] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang),
                                              torch.zeros_like(mag)], -1)
        self._actions[env_ids] = 0.0
        self._prev[env_ids] = 0.0
        if hasattr(self, "vehicle"):
            self.vehicle.reset(env_ids)

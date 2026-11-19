"""Vehicle: thrusters, servos, sensors and water forces on one articulation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from isaacswim.actuators.base import Actuator
from isaacswim.control import VehicleState
from isaacswim.isaaclab import _lab
from isaacswim.torch_math import quat_apply_inverse

__all__ = ["Vehicle"]


class Vehicle:
    def __init__(self, asset, *, hydro=None, actuators: Iterable[Actuator] = (),
                 sensors: Mapping[str, Any] | None = None,
                 seabed_z: float | None = None, step_current: bool = True,
                 generator: torch.Generator | None = None):
        self.asset = asset
        self.hydro = hydro
        self.seabed_z = seabed_z
        self.step_current = step_current
        self.generator = generator
        self.time = 0.0

        try:
            self.body_names = list(asset.body_names)
            self.joint_names = list(asset.joint_names)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "the articulation's body/joint names are not available yet.  Build "
                "the Vehicle after the simulation has been reset.") from exc

        data = _lab.data(asset)
        self.num_envs = int(data.root_pos_w.shape[0])
        self.device = str(data.root_pos_w.device)
        self.dtype = data.root_pos_w.dtype

        self.actuators: dict[str, Actuator] = {}
        body_com = self._body_com()
        default_q = getattr(data, "default_joint_pos", None)
        for act in actuators:
            if act.name in self.actuators:
                raise ValueError(f"two actuators are both named {act.name!r}")
            act.bind(body_names=self.body_names, joint_names=self.joint_names,
                     num_envs=self.num_envs, device=self.device,
                     body_com=body_com, default_joint_pos=default_q)
            self.actuators[act.name] = act

        self.sensors = dict(sensors or {})
        for name, s in self.sensors.items():
            if not hasattr(s, "measure"):
                raise TypeError(f"sensor {name!r} ({type(s).__name__}) has no "
                                "measure(state) method")

        self._write_wrench = self._resolve_writer(asset)

    def _body_com(self):
        view = _lab.physx_view(self.asset)
        if view is not None and hasattr(view, "get_coms"):
            coms = _lab.view_coms(self.asset)
            return coms.reshape(coms.shape[0], -1, 7)[0, :, :3]
        for attr in ("com_pos_b", "body_com_pos_b"):
            com = getattr(_lab.data(self.asset), attr, None)
            if com is not None:
                return com[0].detach().cpu()
        return None

    def state(self) -> VehicleState:
        d = _lab.data(self.asset)
        pos, quat = d.root_pos_w, d.root_quat_w
        altitude = None if self.seabed_z is None else pos[:, 2] - self.seabed_z
        current_b = None
        fn = getattr(getattr(self.hydro, "cfg", None), "current_fn", None)
        if fn is not None:
            current_b = quat_apply_inverse(quat, fn(pos))
        elif self.hydro is not None:
            current_w = torch.as_tensor(self.hydro.cfg.medium.current,
                                        device=pos.device, dtype=pos.dtype).expand_as(pos)
            current_b = quat_apply_inverse(quat, current_w)
        return VehicleState(
            pos_w=pos, quat_w=quat,
            lin_vel_b=d.root_lin_vel_b, ang_vel_b=d.root_ang_vel_b,
            joint_pos=d.joint_pos, joint_vel=d.joint_vel,
            joint_names=self.joint_names, body_names=self.body_names,
            time=self.time, altitude=altitude, current_b=current_b,
        )

    def observe(self, state: VehicleState | None = None) -> dict[str, Any]:
        state = state or self.state()
        obs: dict[str, Any] = {"time": state.time}
        for name, sensor in self.sensors.items():
            out = sensor.measure(state, generator=self.generator)
            if isinstance(out, tuple):
                obs[name], obs[f"{name}/valid"] = out
            else:
                obs[name] = out
        return obs

    def command(self, setpoints: Mapping[str, Any]) -> None:
        for name, value in setpoints.items():
            if name not in self.actuators:
                raise KeyError(f"no actuator named {name!r}; this vehicle has "
                               f"{list(self.actuators)}")
            self.actuators[name].set_setpoint(value)

    def control(self, controller, dt: float) -> dict[str, Any]:
        obs = self.observe()
        self.command(controller(obs, dt) or {})
        return obs

    def apply(self, dt: float, add: tuple[torch.Tensor, torch.Tensor] | None = None):
        E, B = self.num_envs, len(self.body_names)
        F = torch.zeros(E, B, 3, device=self.device, dtype=self.dtype)
        T = torch.zeros_like(F)
        needs_state = any(getattr(a, "needs_state", False)
                          for a in self.actuators.values())
        state = self.state() if needs_state else None

        for act in self.actuators.values():
            out = act.wrench(dt, state)
            if out is not None:
                b, f, t = out
                F[:, b] += f.to(self.dtype)
                T[:, b] += t.to(self.dtype)
        if add is not None:
            F = F + add[0]
            T = T + add[1]

        if self.hydro is not None:
            F, T = self.hydro.apply(add=(F, T), dt=dt)
        else:
            self._write_wrench(F, T)

        for act in self.actuators.values():
            out = act.joint_targets(dt, state)
            if out is not None:
                idx, q = out
                _lab.set_joint_position_target(self.asset, q.to(self.dtype), joint_ids=idx)

        cfg = getattr(self.hydro, "cfg", None)
        for fn in (getattr(cfg, "current_fn", None), getattr(cfg, "surface_fn", None)):
            if self.step_current and fn is not None and hasattr(fn, "step"):
                fn.step(dt)

        self.asset.write_data_to_sim()
        self.time += dt
        return F, T

    def update(self, dt: float) -> None:
        self.asset.update(dt)

    def reset(self, env_ids=None, controller=None) -> None:
        for act in self.actuators.values():
            act.reset(env_ids)
        if self.hydro is not None and hasattr(self.hydro, "reset"):
            self.hydro.reset(env_ids)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset(env_ids)
        if env_ids is None:
            self.time = 0.0

    @staticmethod
    def _resolve_writer(asset):
        return lambda F, T: _lab.write_wrench(asset, F, T)

    def __repr__(self) -> str:
        return (f"Vehicle({self.num_envs} envs, actuators={list(self.actuators)}, "
                f"sensors={list(self.sensors)}, "
                f"{'wet' if self.hydro is not None else 'dry'})")

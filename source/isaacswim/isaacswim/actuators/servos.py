from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch

from isaacswim.actuators.base import Actuator

__all__ = ["JointServos"]


class JointServos(Actuator):
    def __init__(self, joints: Sequence[str], *, rate=None, name: str = "arm"):
        super().__init__(name)
        if isinstance(joints, str):
            joints = [joints]
        self.patterns = list(joints)
        self.rate_spec = rate

    def bind(self, *, body_names, joint_names, num_envs, device, body_com=None,
             default_joint_pos=None) -> None:
        super().bind(body_names=body_names, joint_names=joint_names,
                     num_envs=num_envs, device=device)
        names = list(joint_names)
        idx: list[int] = []
        for pat in self.patterns:
            hits = [i for i, n in enumerate(names) if re.fullmatch(pat, n)]
            if not hits:
                raise ValueError(f"{self.name}: {pat!r} matches no joint; the "
                                 f"articulation has {names}")
            idx += [i for i in hits if i not in idx]
        self.joint_indices = torch.tensor(idx, dtype=torch.long, device=self.device)
        self.joint_names = [names[i] for i in idx]
        self._col = {n: k for k, n in enumerate(self.joint_names)}
        self.size = len(idx)

        if self.rate_spec is None:
            rate = torch.full((self.size,), float("inf"))
        else:
            rate = torch.as_tensor(self.rate_spec, dtype=torch.float32).reshape(-1)
            if rate.numel() == 1:
                rate = rate.expand(self.size)
            if rate.numel() != self.size:
                raise ValueError(f"{self.name}: {rate.numel()} rates for "
                                 f"{self.size} joints")
        self.rate = rate.to(self.device, torch.float32).clone()

        if default_joint_pos is not None:
            start = torch.as_tensor(default_joint_pos, dtype=torch.float32,
                                    device=self.device)[:, self.joint_indices]
        else:
            start = torch.zeros(num_envs, self.size, device=self.device)
        self._default = start.clone()
        self.target = start.clone()
        self.setpoint = start.double()

    def reset(self, env_ids=None) -> None:
        if not self._bound:
            return
        ids = slice(None) if env_ids is None else env_ids
        self.target[ids] = self._default[ids]
        self.setpoint[ids] = self._default[ids].double()

    def set_setpoint(self, value) -> None:
        self._require_bound()
        if isinstance(value, Mapping):
            for joint, v in value.items():
                if joint not in self._col:
                    raise KeyError(f"{self.name}: {joint!r} is not in this group "
                                   f"({self.joint_names})")
                self.setpoint[:, self._col[joint]] = float(v)
            return
        self.setpoint = self._as_batch(value, self.size, torch.float64).clone()

    @property
    def at_setpoint(self) -> torch.Tensor:
        return (self.target == self.setpoint.float()).all(dim=1)

    def joint_targets(self, dt: float, state=None):
        self._require_bound()
        step = self.rate.double() * dt
        have = self.target.double()
        want = self.setpoint
        err = want - have
        close = err.abs() <= step
        self.target = torch.where(close, want, have + torch.sign(err) * step).float()
        return self.joint_indices, self.target

    def __repr__(self) -> str:
        n = self.size if self._bound else "?"
        return f"JointServos({self.name!r}, {n} joints, rate={self.rate_spec})"

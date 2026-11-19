from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["Actuator"]


class Actuator:
    size: int = 0
    needs_state: bool = False

    def __init__(self, name: str):
        self.name = name
        self.num_envs = 0
        self.device = "cpu"
        self._bound = False

    def bind(self, *, body_names: Sequence[str], joint_names: Sequence[str],
             num_envs: int, device: str, body_com=None,
             default_joint_pos=None) -> None:
        self.num_envs = int(num_envs)
        self.device = str(device)
        self._bound = True

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        pass

    def set_setpoint(self, value) -> None:
        raise NotImplementedError(f"{type(self).__name__} takes no setpoint")

    def wrench(self, dt: float, state) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        return None

    def joint_targets(self, dt: float, state) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError(
                f"actuator {self.name!r} is not bound yet.  Add it to a Vehicle, "
                "which binds it once the simulation has been reset and the "
                "articulation's body and joint names exist.")

    def _as_batch(self, value, width: int, dtype=torch.float32) -> torch.Tensor:
        t = torch.as_tensor(value, dtype=dtype, device=self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        if t.shape != (self.num_envs, width):
            raise ValueError(
                f"{self.name}: setpoint has shape {tuple(t.shape)}, expected "
                f"({self.num_envs}, {width}) or ({width},)")
        return t

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, size={self.size})"

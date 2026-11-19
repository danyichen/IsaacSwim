from __future__ import annotations

import torch

from isaacswim.isaaclab import _lab
from isaacswim.oceansim_render import OceanSimWater, apply_underwater

__all__ = ["UnderwaterCamera"]


class UnderwaterCamera:
    def __init__(self, camera, water: OceanSimWater | None = None, *,
                 rgb_key: str = "rgb", range_key: str = "distance_to_image_plane",
                 max_range: float | None = 1000.0, model_device: str | None = None):
        self.camera = camera
        self.water = water if water is not None else OceanSimWater()
        self.rgb_key = rgb_key
        self.range_key = range_key
        self.max_range = max_range
        if model_device is None:
            model_device = "cuda" if torch.cuda.is_available() else None
        self.model_device = model_device
        self._checked = False

    def set_water(self, water: OceanSimWater) -> None:
        self.water = water

    def _check(self, data):
        if self._checked:
            return
        have = set(getattr(data, "output", {}).keys())
        missing = [k for k in (self.rgb_key, self.range_key) if k not in have]
        if missing:
            raise KeyError(
                f"the camera does not produce {missing}; configure it with "
                f'data_types=["{self.rgb_key}", "{self.range_key}"].  Available: '
                f"{sorted(have)}")
        self._checked = True

    def rgb(self) -> torch.Tensor:
        data = self.camera.data
        self._check(data)
        rgb = _lab.to_torch(data.output[self.rgb_key])
        rng = _lab.to_torch(data.output[self.range_key])
        alpha = rgb[..., 3:] if rgb.shape[-1] == 4 else None
        dev = rgb.device
        out = rgb[..., :3]
        if self.model_device is not None:
            out, rng = out.to(self.model_device), rng.to(self.model_device)
        out = apply_underwater(out, rng, self.water, max_range=self.max_range).to(dev)
        return out if alpha is None else torch.cat([out, alpha], dim=-1)

    @property
    def visibility_range(self) -> float:
        return self.water.visibility_range()

    def __repr__(self) -> str:
        return f"UnderwaterCamera({self.water!r})"

"""Buoyancy trim: the foam volume and position that make a vehicle neutral and level."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["TrimVolume", "solve_trim", "displaced_volume"]


@dataclass
class TrimVolume:
    semiaxes: tuple[float, float, float]
    z: float
    max_volume: float | None = None
    drag: bool = False


def _volume_shape(FluidShape, body, vol: TrimVolume, pos, volume):
    kw = {} if vol.drag else dict(coef=np.zeros(5), virtual_mass=np.zeros(3),
                                  virtual_inertia=np.zeros(3))
    return FluidShape(body=body, semiaxes=vol.semiaxes, pos=pos, volume=volume, **kw)


def displaced_volume(volumes: Sequence[float], positions) -> tuple[float, np.ndarray]:
    vols = [float(v) for v in volumes]
    total = sum(vols)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cob = (pos * np.array(vols)[:, None]).sum(0) / total
    return total, cob


def solve_trim(*, geom_volume: float, geom_cob, total_mass: float, com,
               density: float, foam: TrimVolume, makeup: TrimVolume | None = None,
               body: str = "base_link", ballast=None, level_trim: bool = True):
    from isaacswim.shapes import FluidShape

    geom_cob = np.asarray(geom_cob, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)

    want_volume = total_mass / density
    deficit = want_volume - geom_volume
    if deficit <= 0.0:
        raise ValueError(
            f"the geometry already displaces {geom_volume * 1e3:.2f} L against "
            f"{want_volume * 1e3:.2f} L for neutral; there is no room for foam")
    foam_volume = deficit if foam.max_volume is None else min(foam.max_volume, deficit)
    makeup_volume = deficit - foam_volume
    if makeup_volume > 0.0 and makeup is None:
        raise ValueError(
            f"{makeup_volume * 1e3:.2f} L is left after the foam cap and there is "
            "no makeup volume to put it in")

    makeup_shape = None
    makeup_moment = np.zeros(3)
    if makeup_volume > 0.0:
        makeup_shape = _volume_shape(FluidShape, body, makeup, (0.0, 0.0, makeup.z),
                                     makeup_volume)
        makeup_moment = np.array(makeup_shape.pos, dtype=float) * makeup_volume

    fixed_moment = geom_cob[:2] * geom_volume + makeup_moment[:2]
    if ballast is not None:
        foam_xy = np.asarray(ballast, dtype=float)[:2]
    elif level_trim:
        foam_xy = (com[:2] * want_volume - fixed_moment) / foam_volume
    else:
        foam_xy = np.zeros(2)

    foam_shape = _volume_shape(FluidShape, body, foam, (foam_xy[0], foam_xy[1], foam.z),
                               foam_volume)
    shapes = ([makeup_shape] if makeup_shape is not None else []) + [foam_shape]

    cob = (geom_cob * geom_volume + makeup_moment
           + np.array(foam_shape.pos, dtype=float) * foam_volume) / want_volume

    info = {
        "want_volume": want_volume, "geom_volume": geom_volume,
        "foam_volume": foam_volume, "makeup_volume": makeup_volume,
        "geom_cob": geom_cob, "cob": cob, "vehicle_com": com, "bg": cob - com,
        "foam_pos": np.array(foam_shape.pos),
        "ballast": np.array([foam_shape.pos[0], foam_shape.pos[1], 0.0]),
    }
    return shapes, info

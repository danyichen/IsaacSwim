"""Water properties: density, viscosity, gravity, current and the free surface."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Medium"]


@dataclass
class Medium:
    density: float = 997.0
    viscosity: float = 1.0016e-3
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_height: float | None = None

    @staticmethod
    def fresh_water(temperature_c: float = 25.0) -> Medium:
        table = {20.0: (998.2, 1.002e-3), 25.0: (997.0, 0.890e-3)}
        rho, mu = table.get(temperature_c, table[25.0])
        return Medium(density=rho, viscosity=mu)

    @staticmethod
    def sea_water() -> Medium:
        return Medium(density=1025.0, viscosity=1.22e-3)

    @staticmethod
    def air() -> Medium:
        return Medium(density=1.204, viscosity=1.81e-5)

    @staticmethod
    def mujoco_default() -> Medium:
        return Medium(density=1000.0, viscosity=0.001)

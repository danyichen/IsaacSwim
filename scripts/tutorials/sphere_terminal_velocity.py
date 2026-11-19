"""The fluid model without Isaac Sim: a sinking sphere settles at its terminal velocity."""

from __future__ import annotations

import numpy as np
import torch

from isaacswim import FluidShape, HydroWrench, Medium

torch.set_default_dtype(torch.float64)

R = 0.05
BODY_DENSITY = 2700.0
MEDIUM = Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, -9.81))
DT = 1e-3
STEPS = 20000


def main() -> None:
    volume = 4.0 / 3.0 * np.pi * R ** 3
    mass = BODY_DENSITY * volume

    shapes = [FluidShape.sphere("ball", R)]
    hydro = HydroWrench(shapes, ["ball"], MEDIUM, dtype=torch.float64)

    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    pos = torch.zeros(1, 1, 3)
    vel = torch.zeros(1, 1, 3)
    ang = torch.zeros(1, 1, 3)
    g = torch.tensor(MEDIUM.gravity)

    for step in range(STEPS):
        F, _ = hydro.compute(quat, vel, ang, pos_w=pos)
        acc = F[0, 0] / mass + g
        vel[0, 0] += acc * DT
        pos[0, 0] += vel[0, 0] * DT
        if step % 4000 == 0:
            print(f"  t={step * DT:5.1f}s  depth {-pos[0, 0, 2]:6.3f} m  "
                  f"speed {-vel[0, 0, 2]:6.4f} m/s")

    v_sim = -vel[0, 0, 2].item()
    print(f"\nsimulated terminal speed : {v_sim:.4f} m/s")
    print(f"analytic terminal speed  : {analytic_terminal_speed(mass, volume):.4f} m/s")


def analytic_terminal_speed(mass: float, volume: float) -> float:
    rho, mu = MEDIUM.density, MEDIUM.viscosity
    g = abs(MEDIUM.gravity[2])
    a = rho * 0.5 * np.pi * R ** 2
    b = 6.0 * np.pi * mu * R
    c = (mass - rho * volume) * g
    return (-b + np.sqrt(b * b + 4 * a * c)) / (2 * a)


if __name__ == "__main__":
    main()

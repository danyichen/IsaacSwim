"""Drop a dense ball into water in Isaac Sim and compare its sink rate with theory."""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

from isaacswim.launch import prepare_launch

RADIUS = 0.05
BALL_DENSITY = 2700.0
WATER_DENSITY = 1000.0
VISCOSITY = 1.0e-3
GRAVITY = 9.81
VOLUME = 4.0 / 3.0 * math.pi * RADIUS ** 3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cpu")
args = parser.parse_args()
app = AppLauncher(prepare_launch(args)).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

from isaacswim import FluidShape, Medium  # noqa: E402
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg  # noqa: E402


def terminal_speed() -> float:
    weight = (BALL_DENSITY - WATER_DENSITY) * VOLUME * GRAVITY
    quadratic = WATER_DENSITY * 0.5 * math.pi * RADIUS ** 2
    linear = 6.0 * math.pi * VISCOSITY * RADIUS
    return (-linear + math.sqrt(linear ** 2 + 4.0 * quadratic * weight)) / (2.0 * quadratic)


def main() -> None:
    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args.device, gravity=(0.0, 0.0, -GRAVITY)))
    ball = RigidObject(RigidObjectCfg(
        prim_path="/World/ball",
        spawn=sim_utils.SphereCfg(
            radius=RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_DENSITY * VOLUME),
        ),
    ))
    sim.reset()
    water = Medium(density=WATER_DENSITY, viscosity=VISCOSITY, gravity=(0.0, 0.0, -GRAVITY))
    hydro = Hydrodynamics(
        HydrodynamicsCfg(shapes=[FluidShape.sphere(ball.body_names[0], RADIUS)], medium=water),
        ball,
    )
    for step in range(round(args.seconds / dt)):
        hydro.apply()
        ball.write_data_to_sim()
        sim.step(render=False)
        ball.update(dt)
        if step % 120 == 0:
            print(f"t={step * dt:4.1f} s  sink speed {-ball.data.root_lin_vel_w[0, 2]:.4f} m/s",
                  flush=True)
    speed = float(-ball.data.root_lin_vel_w[0, 2])
    expected = terminal_speed()
    print(f"simulated terminal speed: {speed:.4f} m/s")
    print(f"analytic terminal speed:  {expected:.4f} m/s")
    if abs(speed - expected) > 0.02 * expected:
        raise SystemExit("the ball did not reach the analytic terminal speed")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as exc:
        print(exc, flush=True)
        code = 1
    sys.stdout.flush()
    os._exit(code)

"""Check the built vehicle in air before adding water: structure, mass, free fall and joints."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from isaacswim.assets.bluerov import ARM_HOME, JAWS_USD
from isaacswim.launch import prepare_launch

parser = argparse.ArgumentParser(description="Check the built vehicle in air.")
parser.add_argument("--usd", type=str, default=str(JAWS_USD))
parser.add_argument("--drop-time", type=float, default=0.5, help="seconds of free fall")
parser.add_argument("--spawn-height", type=float, default=20.0,
                    help="high enough that nothing touches the ground during the drop")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
prepare_launch(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

from isaacswim.isaaclab import _lab  # noqa: E402

DT = 1.0 / 200.0
FAILURES: list[str] = []

def joint_limits(robot):
    for attr in ("joint_pos_limits", "joint_limits"):
        limits = getattr(robot.data, attr, None)
        if limits is not None:
            return limits
    raise AttributeError("no joint limit tensor on ArticulationData")


def check(ok: bool, label: str, detail: str) -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<28} {detail}", flush=True)
    if not ok:
        FAILURES.append(label)


usd = Path(args.usd)
if not usd.is_file():
    raise SystemExit(f"{usd} not found -- run scripts/tools/build_bluerov.py first")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu"))

light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)

robot = Articulation(ArticulationCfg(
    prim_path="/World/rov",
    spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, args.spawn_height), joint_pos=ARM_HOME,
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=100.0, damping=10.0,
        ),
    },
))

sim.reset()

print("\n1. structure")
body_names = robot.body_names
joint_names = robot.joint_names
check(robot.is_fixed_base is False, "floating base",
      f"is_fixed_base={robot.is_fixed_base} (must be False for an ROV)")
check(len(body_names) > 1, "bodies", f"{len(body_names)}: {body_names}")
print("         joint order (this is your action index):")
for i, name in enumerate(joint_names):
    lo, hi = joint_limits(robot)[0, i].tolist()
    print(f"           [{i}] {name:<36} [{lo:+.3f}, {hi:+.3f}]")

print("\n2. mass")
masses = _lab.view_masses(robot)[0]
total = float(masses.sum())
check(abs(total - 14.4285) < 0.05, "total mass", f"{total:.4f} kg (URDF declares 14.4285)")
coms = _lab.view_coms(robot).numpy()
base_com = coms.reshape(coms.shape[0], -1, 7)[0, 0, :3]
print(f"         base_link com ({base_com[0]:+.4f} {base_com[1]:+.4f} {base_com[2]:+.4f})")
for name, m in zip(body_names, masses.tolist()):
    print(f"           {name:<38} {m:8.4f} kg")

print(f"\n3. free fall for {args.drop_time:g} s", flush=True)
z0 = float(_lab.data(robot).root_pos_w[0, 2])
_lab.set_joint_position_target(robot, _lab.data(robot).default_joint_pos)
n = int(round(args.drop_time / DT))
for _ in range(n):
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(DT)
z1 = float(_lab.data(robot).root_pos_w[0, 2])

g = abs(float(sim.cfg.gravity[2]))
actual = z0 - z1
vz = float(_lab.data(robot).root_lin_vel_w[0, 2])

g_measured = abs(vz) / (n * DT)
check(abs(g_measured - g) < 0.05 * g, "gravity",
      f"measured {g_measured:.3f} m/s^2 vs {g:.3f}; fell {actual:.4f} m in {n} steps")
check(vz < -0.5 * g * args.drop_time, "actually falling",
      f"vz {vz:+.4f} m/s (a fixed base or a resting collider would sit near zero)")

wx, wy, wz = _lab.data(robot).root_ang_vel_w[0].tolist()
spin = math.sqrt(wx * wx + wy * wy + wz * wz)
check(spin < 1.0, "no tumble",
      f"|omega| {spin:.4f} rad/s -- some spin is expected from the arm settling")

print("\n4. joint drives", flush=True)

state = _lab.data(robot).default_root_state.clone()
state[0, 2] = args.spawn_height
_lab.write_root_state(robot, state)
_lab.write_joint_state(robot, _lab.data(robot).default_joint_pos,
                       torch.zeros_like(_lab.data(robot).default_joint_pos))
robot.reset()

limits = joint_limits(robot)
arm = [i for i, n_ in enumerate(joint_names) if n_.startswith("alpha_")]
if not arm:
    check(False, "arm joints present", "no joint named alpha_* -- wrong USD?")

for i in arm:
    lo, hi = limits[0, i].tolist()
    start = float(_lab.data(robot).joint_pos[0, i])
    span = hi - lo
    target = lo + (0.75 * span if start < 0.5 * (lo + hi) else 0.25 * span)

    vmax = max(float(_lab.data(robot).joint_vel_limits[0, i]) if hasattr(
        robot.data, "joint_vel_limits") else 0.5, 1e-3)
    budget = int((abs(target - start) / vmax + 1.5) / DT)

    cmd = _lab.data(robot).joint_pos.clone()
    cmd[0, i] = target
    for _ in range(budget):
        _lab.set_joint_position_target(robot, cmd)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)

    reached = float(_lab.data(robot).joint_pos[0, i])
    err = abs(reached - target)
    moved = abs(target - start)
    check(err < 0.05 and moved > 0.05, joint_names[i],
          f"{start:+.3f} -> {target:+.3f} ({moved:.3f} rad), reached {reached:+.3f} "
          f"(err {err:.4f} rad in {budget * DT:.1f} s)")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("fix these before adding water -- every one of them mimics a hydro bug.\n")
else:
    print("all checks passed. the rig is sound; now add isaacswim.\n")

sys.stdout.flush()
os._exit(1 if FAILURES else 0)

"""Import an MJCF swimmer into Isaac Sim and compare its shapes and motion with MuJoCo."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

from isaacswim.launch import prepare_launch

parser = argparse.ArgumentParser(description="Compare an MJCF swimmer in MuJoCo and in Isaac Sim.")
parser.add_argument("--mjcf", type=str, default=None)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cpu")
args = parser.parse_args()
app = AppLauncher(prepare_launch(args)).app

import json  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, _lab  # noqa: E402
from isaacswim.mjcf import shapes_from_mjmodel  # noqa: E402
from isaacswim.np_math import quat_apply  # noqa: E402
from isaacswim.usd import shapes_from_articulation_usd  # noqa: E402

OCTAHEDRON = """v 0.12 0 0
v -0.12 0 0
v 0 0.05 0
v 0 -0.05 0
v 0 0 0.03
v 0 0 -0.03
f 1 3 5
f 3 2 5
f 2 4 5
f 4 1 5
f 3 1 6
f 2 3 6
f 4 2 6
f 1 4 6
"""

SWIMMER = """<mujoco model="swimmer">
  <compiler angle="radian" meshdir="."/>
  <option timestep="{dt}" integrator="Euler" density="1000" viscosity="0.0009" gravity="0 0 0"/>
  <asset><mesh name="tail" file="tail.obj"/></asset>
  <worldbody>
    <body name="torso" pos="0 0 0">
      <freejoint/>
      <inertial pos="0.02 0 0" mass="2.0" diaginertia="0.004 0.02 0.021"/>
      <geom name="hull" type="capsule" fromto="-0.15 0 0 0.15 0 0" size="0.05"
            fluidshape="ellipsoid"/>
      <geom name="fin" type="box" pos="0 0 0.06" euler="0 0 0.2" size="0.08 0.005 0.03"
            fluidshape="ellipsoid"/>
      <geom name="nose" type="sphere" pos="0.2 0 0" size="0.04"
            fluidshape="ellipsoid"/>
      <body name="mid" pos="-0.17 0 0">
        <joint name="yaw" type="hinge" axis="0 0 1"/>
        <inertial pos="-0.08 0 0" mass="0.5" diaginertia="0.0003 0.0012 0.0012"/>
        <geom name="mid" type="cylinder" fromto="0 0 0 -0.16 0 0" size="0.035"
              fluidshape="ellipsoid"/>
        <body name="tail" pos="-0.18 0 0">
          <joint name="pitch" type="hinge" axis="0 1 0"/>
          <inertial pos="-0.06 0 0" mass="0.2" diaginertia="0.0001 0.0005 0.0005"/>
          <geom name="tail" type="mesh" mesh="tail" pos="-0.12 0 0" euler="0.3 0 0"
                fluidshape="ellipsoid"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

THRUST = np.array([1.5, 0.0, 0.3])
V0 = np.array([0.3, 0.05, 0.0, 0.0, 0.2, 0.4])


def joint_torque(t: float) -> np.ndarray:
    return np.array([0.15 * np.sin(2 * np.pi * 1.5 * t), 0.05 * np.sin(2 * np.pi * 2.0 * t + 0.5)])


def quadric(shape) -> np.ndarray:
    R = np.array(mujoco_quat_to_mat(shape.quat))
    return R @ np.diag(1.0 / np.asarray(shape.semiaxes) ** 2) @ R.T


def mujoco_quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


work = Path(args.out or tempfile.mkdtemp(prefix="isaacswim_mjcf_")).resolve()
work.mkdir(parents=True, exist_ok=True)
if args.mjcf:
    mjcf = Path(args.mjcf)
else:
    (work / "tail.obj").write_text(OCTAHEDRON)
    mjcf = work / "swimmer.xml"
    mjcf.write_text(SWIMMER.format(dt=args.dt))

model = mujoco.MjModel.from_xml_path(str(mjcf))
ref_shapes, medium = shapes_from_mjmodel(model)
steps = int(round(args.seconds / args.dt))
torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("yaw", "pitch")]
dofs = [int(model.jnt_dofadr[j]) for j in joints]
qadr = [int(model.jnt_qposadr[j]) for j in joints]



def run_mujoco(model) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qvel[0:3] = quat_apply(data.qpos[3:7], V0[:3])
    data.qvel[3:6] = V0[3:]
    mujoco.mj_forward(model, data)
    traj = []
    for k in range(steps):
        traj.append(np.concatenate([data.qpos[0:7], data.qpos[qadr]]))
        data.xfrc_applied[torso, :3] = quat_apply(data.xquat[torso], THRUST)
        data.qfrc_applied[dofs] = joint_torque(k * args.dt)
        mujoco.mj_step(model, data)
    return np.array(traj)


mj_traj = run_mujoco(model)
model.opt.density = 0.0
model.opt.viscosity = 0.0
dry_traj = run_mujoco(model)

usd_dir = work / "usd"
convert = dict(asset_path=str(mjcf), usd_dir=str(usd_dir), fix_base=False,
               force_usd_conversion=True)
if _lab.lab_major() < 3:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.mjcf")
    app.update()
    convert["import_inertia_tensor"] = True
converter = MjcfConverter(MjcfConverterCfg(**convert))
from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(converter.usd_path)
top = stage.GetDefaultPrim().GetPath()
roots = [p for p in stage.Traverse(Usd.TraverseInstanceProxies())
         if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
root_path = str(roots[0].GetPath())[len(str(top)):]
del stage
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device="cpu",
                                                          gravity=(0.0, 0.0, 0.0)))
robot = Articulation(ArticulationCfg(
    prim_path="/World/swimmer",
    spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
    articulation_root_prim_path=root_path,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
                                             friction=0.0, armature=0.0,
                                             effort_limit_sim=100.0, velocity_limit_sim=100.0)},
))
sim.reset()

FAIL: list[str] = []


def check(ok: bool, label: str, detail: str) -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<26} {detail}", flush=True)
    if not ok:
        FAIL.append(label)


print("\n1. shapes")
got = shapes_from_articulation_usd(robot)
check(len(got) == len(ref_shapes), "shape count", f"isaac {len(got)}, mujoco {len(ref_shapes)}")
worst = 0.0
for ref in ref_shapes:
    cands = [s for s in got if s.body == ref.body]
    if not cands:
        check(False, f"body {ref.body}", "no ingested shape")
        continue
    Q = quadric(ref)
    errs = [np.abs(quadric(s) - Q).max() / np.abs(Q).max()
            + np.linalg.norm(np.asarray(s.pos) - ref.pos) / max(ref.semiaxes) for s in cands]
    best = cands[int(np.argmin(errs))]
    e = min(errs)
    worst = max(worst, e)
    print(f"         {ref.body:<6} mujoco {np.round(np.sort(ref.semiaxes), 4)} "
          f"isaac {np.round(np.sort(best.semiaxes), 4)} rel {e:.1e}")
    if np.any(np.asarray(best.coef) != np.asarray(ref.coef)):
        check(False, f"coef {ref.body}", f"{best.coef} vs {ref.coef}")
check(worst < 1e-4, "shape geometry", f"worst relative difference {worst:.1e}")

print("\n2. trajectory")
body_names = list(robot.body_names)
joint_names = list(robot.joint_names)
hydro = Hydrodynamics(HydrodynamicsCfg(auto_shapes=True, medium=medium, buoyancy=False,
                                       mesh_buoyancy=False), robot)
ti = body_names.index("torso")
jidx = [joint_names.index(n) for n in ("yaw", "pitch")]
d = _lab.data(robot)
_lab.write_root_pose(robot, torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), link=True)
_lab.write_root_velocity(robot, torch.tensor([V0], dtype=torch.float32), link=True)
zeros = torch.zeros(1, len(joint_names))
_lab.write_joint_state(robot, zeros, zeros)
robot.update(0.0)
nb = len(body_names)
is_traj = []
for k in range(steps):
    t = k * args.dt
    is_traj.append(np.concatenate([d.root_link_pos_w[0].numpy(), d.root_link_quat_w[0].numpy(),
                                   d.joint_pos[0, jidx].numpy()]))
    F = torch.zeros(1, nb, 3)
    T = torch.zeros(1, nb, 3)
    F[0, ti] = torch.tensor(THRUST, dtype=torch.float32)
    hydro.apply(add=(F, T))
    effort = torch.zeros(1, len(joint_names))
    effort[0, jidx] = torch.tensor(joint_torque(t), dtype=torch.float32)
    _lab.set_joint_effort_target(robot, effort)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(args.dt)
is_traj = np.array(is_traj)

pos_err = np.linalg.norm(is_traj[:, :3] - mj_traj[:, :3], axis=1)
path = np.linalg.norm(np.diff(mj_traj[:, :3], axis=0), axis=1).sum()
dq = np.abs(np.sum(is_traj[:, 3:7] * mj_traj[:, 3:7], axis=1)).clip(max=1.0)
ang_err = np.degrees(2 * np.arccos(dq))
jerr = np.degrees(np.abs(is_traj[:, 7:] - mj_traj[:, 7:])).max(0)
jamp = np.degrees(np.abs(mj_traj[:, 7:])).max(0)
dry = np.linalg.norm(dry_traj[-1, :3] - mj_traj[-1, :3])
check(dry > 10 * pos_err[-1], "water matters",
      f"MuJoCo without water ends {dry * 100:.1f} cm away")
check(pos_err[-1] < 0.02 * path, "root position",
      f"{pos_err[-1] * 100:.2f} cm after {path:.2f} m ({100 * pos_err[-1] / path:.2f} %)")
check(ang_err.max() < 3.0, "root attitude", f"max {ang_err.max():.2f} deg")
check(bool(np.all(jerr < 0.05 * jamp + 0.5)), "joint angles",
      f"max error {np.round(jerr, 2)} deg on swings of {np.round(jamp, 1)} deg")
summary = {"steps": steps, "dt": args.dt, "path_m": float(path),
           "final_position_error_m": float(pos_err[-1]),
           "max_attitude_error_deg": float(ang_err.max()),
           "max_joint_error_deg": jerr.tolist(), "joint_swing_deg": jamp.tolist(),
           "worst_shape_difference": float(worst),
           "dry_final_offset_m": float(dry), "failures": FAIL}
(work / "comparison.json").write_text(json.dumps(summary, indent=2))
np.savez(work / "trajectories.npz", mujoco=mj_traj, isaac=is_traj)
print(f"\n{'all checks passed' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}  ({work})")
sys.stdout.flush()
os._exit(1 if FAIL else 0)

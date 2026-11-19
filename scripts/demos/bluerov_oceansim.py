"""A BlueROV2 swimming with OceanSim's camera, imaging sonar, DVL and barometer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from isaacswim.assets.bluerov import JAWS_USD
from isaacswim.launch import prepare_launch

parser = argparse.ArgumentParser(description="OceanSim sensors on a swimming BlueROV2.")
parser.add_argument("--usd", type=str, default=str(JAWS_USD))
parser.add_argument("--out", type=str, required=True,
                    help="where the report, sonar image and camera frame go.  Keep "
                         "it outside a synced folder.")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--no-oceansim", action="store_true",
                    help="IsaacSwim sensors only -- the interface check without "
                         "the OceanSim dependency")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_oceansim
args.headless = True

prepare_launch(args, oceansim=not args.no_oceansim)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

import isaacswim.oceansim as osim  # noqa: E402
from isaacswim import IMU, DopplerVelocityLog, Odometry, PressureDepth  # noqa: E402
from isaacswim.actuators import JointServos, ThrusterArray, Thrusters  # noqa: E402
from isaacswim.assets.bluerov import ARM_HOME, HoldPose, StationKeeper, build_hydro  # noqa: E402
from isaacswim.isaaclab import Vehicle  # noqa: E402

DT = 1.0 / 120.0
DECIMATION = 4
SURFACE_Z = 3.0
SEABED_Z = -1.2
ROCK = (1.9, 0.0, SEABED_Z + 0.2)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
usd = Path(args.usd)
urdf = usd.with_suffix(".urdf")

use_oceansim = not args.no_oceansim
if use_oceansim and not osim.enable():
    raise SystemExit("OceanSim could not be enabled.\n" + osim.OCEANSIM_INSTALL_HINT)

sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=DT, device=args.device, gravity=(0.0, 0.0, -9.81),
                            enable_scene_query_support=True))
dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.62, 0.78, 0.92))
dome.func("/World/DomeLight", dome)
seabed = sim_utils.GroundPlaneCfg(color=(0.28, 0.26, 0.22))
seabed.func("/World/seabed", seabed, translation=(0.0, 0.0, SEABED_Z))
rock = sim_utils.CuboidCfg(
    size=(0.4, 0.4, 0.4), collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.45, 0.4)))
rock.func("/World/rock", rock, translation=ROCK)

robot = Articulation(ArticulationCfg(
    prim_path="/World/rov",
    spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=ARM_HOME),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                          stiffness=200.0, damping=30.0)},
))


def find_prim(root: str, name: str) -> str:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    from pxr import Usd
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise SystemExit(f"no prim called {name!r} under {root}")


def label_meshes(root: str, reflectivity: float) -> int:
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Gprim):
            osim.set_sonar_reflectivity(str(prim.GetPath()), reflectivity)
            n += 1
    return n


base_prim = find_prim("/World/rov", "base_link")
water = None
oceansim_sensors = {}
if use_oceansim:
    labelled = label_meshes("/World/seabed", 1.0) + label_meshes("/World/rock", 2.0)
    print(f"sonar: labelled {labelled} meshes; mounting on {base_prim}", flush=True)
    oceansim_sensors = {
        "sonar": osim.Sonar(base_prim, pitch_deg=30.0, max_range=4.0,
                            hori_res=1000, device=args.device),
        "dvl_os": osim.DVL(base_prim, max_range=10.0, device=args.device),
        "cam": osim.Camera(base_prim, resolution=(320, 240), device=args.device),
    }

sim.reset()
device = sim.device

hydro, _ = build_hydro(robot, urdf, surface_z=SURFACE_Z, seabed_z=SEABED_Z, device=device)
water = hydro.cfg.medium
if use_oceansim:
    oceansim_sensors["baro_os"] = osim.Barometer(base_prim, water, as_depth=True,
                                                 device=device)

vehicle = Vehicle(
    robot, hydro=hydro, seabed_z=SEABED_Z,
    actuators=[
        Thrusters(ThrusterArray(usd.parent / "thrusters.json"), body="base_link"),
        JointServos(["alpha_axis_.*"], rate=0.5, name="arm"),
    ],
    sensors={
        "odom": Odometry(),
        "imu": IMU(),
        "depth": PressureDepth(surface_height=SURFACE_Z, density=water.density,
                               sigma=0.0),
        "dvl": DopplerVelocityLog(sigma=0.0, scale_error=0.0),
        **oceansim_sensors,
    },
)
print(f"\n{vehicle}", flush=True)
controller = HoldPose(StationKeeper(), target_pos=(0.0, 0.0, 0.0))

steps = int(args.seconds / DT)
rows = []
sonar_img = cam_img = None
for step in range(steps):
    t = step * DT
    if step % DECIMATION == 0:
        controller.target_pos = np.array([min(max(t - 3.0, 0.0) * 0.1, 1.0), 0.0, 0.0])
        truth = vehicle.state()
        obs = vehicle.observe(truth)
        vehicle.command(controller(obs, DT * DECIMATION))

        row = {"t": t, "v_true": truth.lin_vel_b[0].tolist(),
               "dvl": obs["dvl"][0].tolist(), "depth": float(obs["depth"][0]),
               "pos": truth.pos_w[0].tolist()}
        if use_oceansim and step > 0:
            row.update(dvl_os=obs["dvl_os"][0].tolist(),
                       dvl_os_valid=bool(obs["dvl_os/valid"][0]),
                       baro_os=float(obs["baro_os"][0]),
                       sonar_valid=bool(obs["sonar/valid"][0]),
                       sonar_peak=float(obs["sonar"].max()))
            if obs["sonar/valid"][0]:
                sonar_img = obs["sonar"][0].cpu().numpy()
            cam_img = obs["cam"][0].cpu().numpy()
        rows.append(row)

    vehicle.apply(DT)
    sim.step(render=use_oceansim and (step + 1) % DECIMATION == 0)
    vehicle.update(DT)

def err(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


report = {
    "seconds": args.seconds,
    "final_pos": rows[-1]["pos"],
    "station_err_max_m": max(err(r["pos"], [min(max(r["t"] - 3.0, 0) * 0.1, 1.0), 0, 0])
                             for r in rows if r["t"] > 1.0),
    "isaacswim_dvl_err_max": max(err(r["dvl"], r["v_true"]) for r in rows),
    "depth_isaacswim": rows[-1]["depth"],
    "depth_true": SURFACE_Z - rows[-1]["pos"][2],
}
if use_oceansim:
    os_rows = [r for r in rows if "dvl_os" in r]
    valid = [r for r in os_rows if r["dvl_os_valid"]]
    report.update({
        "oceansim_dvl_valid_frac": len(valid) / max(len(os_rows), 1),
        "oceansim_dvl_err_mean": float(np.mean([err(r["dvl_os"], r["v_true"])
                                                for r in valid])) if valid else None,
        "oceansim_dvl_err_moving_max": max((err(r["dvl_os"], r["v_true"])
                                            for r in valid if r["t"] > 3.5),
                                           default=None),
        "baro_vs_pressure_depth_max_m": max(abs(r["baro_os"] - r["depth"])
                                            for r in os_rows),
        "sonar_valid_frac": sum(r["sonar_valid"] for r in os_rows) / max(len(os_rows), 1),
        "sonar_peak": max(r["sonar_peak"] for r in os_rows),
        "camera_mean": float(cam_img.mean()) if cam_img is not None else None,
    })
    import imageio.v2 as imageio
    if sonar_img is not None:
        s = sonar_img / max(float(sonar_img.max()), 1e-9)
        imageio.imwrite(out / "sonar.png", (np.clip(s, 0, 1) * 255).astype(np.uint8))
    if cam_img is not None:
        imageio.imwrite(out / "camera.png", cam_img[..., :3].astype(np.uint8))

print("\n" + json.dumps(report, indent=2), flush=True)
(out / "oceansim_sensors.json").write_text(json.dumps(report, indent=2))
(out / "trace.json").write_text(json.dumps(rows))
sys.stdout.flush()
os._exit(0)

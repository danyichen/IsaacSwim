"""OceanSim waves, sensors and ROS 2 topics alongside IsaacSwim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from isaacswim.assets.bluerov import JAWS_USD
from isaacswim.launch import prepare_launch

parser = argparse.ArgumentParser(description="OceanSim waves, sensors and ROS 2 with IsaacSwim.")
parser.add_argument("--usd", type=str, default=str(JAWS_USD))
parser.add_argument("--out", type=str, required=True,
                    help="where the report and images go.  Keep it outside a synced "
                         "folder.")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--no-ros", action="store_true", help="skip ROS 2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
use_ros = not args.no_ros

prepare_launch(args, oceansim=True, ros2=use_ros)
simulation_app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402

import isaacswim.oceansim as osim  # noqa: E402
from isaacswim import FluidShape, Medium, Odometry  # noqa: E402
from isaacswim.actuators import ThrusterArray, Thrusters  # noqa: E402
from isaacswim.assets.bluerov import ARM_HOME, HoldPose, StationKeeper, build_hydro  # noqa: E402
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, Vehicle, _lab  # noqa: E402

DT = 1.0 / 120.0
DECIMATION = 4
SURFACE_Z = 3.0
SEABED_Z = -1.2
HOLD_Z = SURFACE_Z - 1.5
BUOY_R, BUOY_MASS = 0.25, 15.0
CMD = (4.0, 8.0, 0.15)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
usd = Path(args.usd)
if not osim.enable():
    raise SystemExit("OceanSim could not be enabled.\n" + osim.OCEANSIM_INSTALL_HINT)

sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=DT, device=args.device, gravity=(0.0, 0.0, -9.81)))
dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.62, 0.78, 0.92))
dome.func("/World/DomeLight", dome)
seabed = sim_utils.GroundPlaneCfg(color=(0.28, 0.26, 0.22))
seabed.func("/World/seabed", seabed, translation=(0.0, 0.0, SEABED_Z))
rock = sim_utils.CuboidCfg(
    size=(0.4, 0.4, 0.4), collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.45, 0.4)))
rock.func("/World/rock", rock, translation=(1.9, 0.0, SEABED_Z + 0.2))

waves = osim.Waves("/World/WaterSurface", height=SURFACE_Z, size=(40.0, 40.0),
                   dim=(160, 160), amplitude=0.6, windSpeed=7.0)

robot = Articulation(ArticulationCfg(
    prim_path="/World/rov",
    spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, HOLD_Z), joint_pos=ARM_HOME),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                          stiffness=200.0, damping=30.0)},
))
buoy = RigidObject(RigidObjectCfg(
    prim_path="/World/buoy",
    spawn=sim_utils.SphereCfg(
        radius=BUOY_R, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=BUOY_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05))),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.8, SURFACE_Z)),
))


def find_prim(root: str, name: str) -> str:
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise SystemExit(f"no prim called {name!r} under {root}")


def label_meshes(root: str, reflectivity: float) -> None:
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Gprim):
            osim.set_sonar_reflectivity(str(prim.GetPath()), reflectivity)


base_prim = find_prim("/World/rov", "base_link")
label_meshes("/World/seabed", 1.0)
label_meshes("/World/rock", 2.0)
ros = osim.ros_publishers("isaacswim", camera=True, sonar=True, baro=True) if use_ros \
    else None
oceansim_sensors = {
    "sonar": osim.Sonar(base_prim, pitch_deg=30.0, max_range=4.0, hori_res=1000,
                        device=args.device, ros=ros),
    "cam": osim.Camera(base_prim, resolution=(320, 240), device=args.device, ros=ros),
}

cmd_vel = None
if use_ros:
    from isaacswim.ros2 import CmdVel, publish_vehicle
    try:
        publish_vehicle("/World/rov", base_prim)
        cmd_vel = CmdVel("/cmd_vel")
    except RuntimeError as e:
        print(f"[ecosystem] {e}\n  (or run with --no-ros)", file=sys.stderr, flush=True)
        os._exit(1)

sim.reset()
device = sim.device

hydro, _ = build_hydro(robot, usd.with_suffix(".urdf"), surface_z=SURFACE_Z,
                       seabed_z=SEABED_Z, device=device, surface_fn=waves)
water = hydro.cfg.medium
oceansim_sensors["baro"] = osim.Barometer(base_prim, water, as_depth=True,
                                          device=device, ros=ros)
vehicle = Vehicle(
    robot, hydro=hydro, seabed_z=SEABED_Z,
    actuators=[Thrusters(ThrusterArray(usd.parent / "thrusters.json"), body="base_link")],
    sensors={"odom": Odometry(), **oceansim_sensors},
)
buoy_hydro = Hydrodynamics(HydrodynamicsCfg(
    shapes=[FluidShape.sphere(buoy.body_names[0], BUOY_R)],
    medium=Medium(density=water.density, viscosity=water.viscosity,
                  gravity=water.gravity),
    surface_fn=waves), buoy)
controller = HoldPose(StationKeeper(), target_pos=(0.0, 0.0, HOLD_Z))

node = None
if use_ros:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry as OdomMsg
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import FluidPressure, Image, JointState
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = rclpy.create_node("isaacswim_ecosystem_check")
    counts, last = {}, {}

    def listen(topic, kind):
        def cb(msg):
            counts[topic] = counts.get(topic, 0) + 1
            last[topic] = msg
        node.create_subscription(kind, topic, cb, qos_profile_sensor_data)

    for topic, kind in [("/clock", Clock), ("/tf", TFMessage),
                        ("/joint_states", JointState), ("/odom", OdomMsg),
                        ("/isaacswim/sonar_image", Image), ("/isaacswim/baro", FluidPressure),
                        ("/isaacswim/rgb", Image)]:
        listen(topic, kind)
    twist_pub = node.create_publisher(Twist, "/cmd_vel", 10)

rows = []
sonar_img = cam_img = None
target = np.array([0.0, 0.0, HOLD_Z])
for step in range(int(args.seconds / DT)):
    t = step * DT
    if step % DECIMATION == 0:
        if node is not None:
            msg = Twist()
            msg.linear.x = CMD[2] if CMD[0] <= t < CMD[1] else 0.0
            twist_pub.publish(msg)
            for _ in range(32):
                rclpy.spin_once(node, timeout_sec=0.0)
            lin, _ = cmd_vel.read()
            target = target + lin * DT * DECIMATION
        controller.target_pos = target
        truth = vehicle.state()
        obs = vehicle.observe(truth)
        vehicle.command(controller(obs, DT * DECIMATION))
        bpos = _lab.data(buoy).root_pos_w[0].cpu().numpy()
        rows.append({"t": t, "rov": truth.pos_w[0].tolist(), "target": target.tolist(),
                     "buoy_z": float(bpos[2]),
                     "wave_at_buoy": float(waves.height_at(bpos[None, :2])[0]),
                     "baro": float(obs["baro"][0]),
                     "sonar_valid": bool(obs["sonar/valid"][0])})
        cam_img = obs["cam"][0].cpu().numpy()
        if obs["sonar/valid"][0]:
            sonar_img = obs["sonar"][0].cpu().numpy()

    vehicle.apply(DT)
    buoy_hydro.apply()
    buoy.write_data_to_sim()
    sim.step(render=(step + 1) % DECIMATION == 0)
    vehicle.update(DT)
    buoy.update(DT)
    if use_ros and (step + 1) % DECIMATION == 0:
        from isaacswim import ros2
        ros2.update()

settled = [r for r in rows if r["t"] > 2.0]
heave = np.array([r["buoy_z"] for r in settled])
wave = np.array([r["wave_at_buoy"] for r in settled])
moved = [r for r in rows if r["t"] >= CMD[1] + 2.0]
report = {
    "buoy_heave_range_m": float(np.ptp(heave)),
    "wave_range_at_buoy_m": float(np.ptp(wave)),
    "buoy_heave_vs_wave_corr": float(np.corrcoef(heave, wave)[0, 1]),
    "rov_depth_hold_err_max_m": max(abs(r["rov"][2] - r["target"][2]) for r in settled),
    "rov_x_final_m": rows[-1]["rov"][0],
    "baro_depth_final_m": rows[-1]["baro"],
    "sonar_valid_frac": float(np.mean([r["sonar_valid"] for r in rows[1:]])),
}
if node is not None:
    report["ros_messages"] = counts
    report["cmd_vel_target_x_m"] = moved[-1]["target"][0] if moved else None
    if "/clock" in last:
        stamp = last["/clock"].clock
        report["ros_clock_final_s"] = stamp.sec + stamp.nanosec * 1e-9
    if "/tf" in last:
        report["tf_transform_count"] = len(last["/tf"].transforms)
    if "/odom" in last:
        report["odom_final_x_m"] = last["/odom"].pose.pose.position.x
failures = []
if report["rov_depth_hold_err_max_m"] > 0.2:
    failures.append("vehicle did not hold depth within 0.2 m")
if report["buoy_heave_vs_wave_corr"] < 0.9:
    failures.append("buoy heave did not track the wave surface")
if report["sonar_valid_frac"] < 0.9:
    failures.append("sonar returned too few valid scans")
if node is not None:
    if abs(report.get("ros_clock_final_s", 0.0) - args.seconds) > 0.1:
        failures.append("ROS clock does not match simulated time")
    if report.get("tf_transform_count", 0) == 0:
        failures.append("TF messages contain no transforms")
    required = {"/clock", "/tf", "/joint_states", "/odom", "/isaacswim/sonar_image",
                "/isaacswim/baro", "/isaacswim/rgb"}
    missing = sorted(topic for topic in required if counts.get(topic, 0) == 0)
    if missing:
        failures.append(f"no ROS messages on {missing}")
    if args.seconds >= CMD[1] + 2.0:
        expected = CMD[2] * (CMD[1] - CMD[0])
        if abs(report["cmd_vel_target_x_m"] - expected) > 0.1:
            failures.append("cmd_vel did not reach the controller")
report["validation_failures"] = failures
print("\n" + json.dumps(report, indent=2), flush=True)
(out / "ecosystem.json").write_text(json.dumps(report, indent=2))
(out / "trace.json").write_text(json.dumps(rows))

import imageio.v2 as imageio  # noqa: E402

imageio.imwrite(out / "camera.png", cam_img[..., :3].astype(np.uint8))
if sonar_img is not None:
    s = sonar_img / max(float(sonar_img.max()), 1e-9)
    imageio.imwrite(out / "sonar.png", (np.clip(s, 0, 1) * 255).astype(np.uint8))
sys.stdout.flush()
os._exit(int(bool(failures)))

# OceanSim and ROS 2

| | From | IsaacSwim's part |
|---|---|---|
| Waves, caustics | OceanSim `WaterSurface` | `Waves` cuts buoyancy with OceanSim's wave function |
| Sonar, DVL, barometer, camera | OceanSim sensors | Adapters mount them on a `Vehicle` |
| Sensor topics | OceanSim `*_ROS` classes | `ros=` on the adapters |
| `/clock`, `/tf`, `/joint_states`, `/odom`, `/cmd_vel` | Isaac Sim ROS 2 bridge | `isaacswim.ros2` wires them to a vehicle |

[`scripts/demos/ecosystem.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/demos/ecosystem.py) uses all of them:
a vehicle holds station under waves beside a floating buoy, OceanSim's sensors
publish over ROS 2, and a ROS 2 node steers the vehicle through `/cmd_vel`.

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/demos/ecosystem.py --out runs/ecosystem
```

## Waves

```python
from isaacswim import oceansim as osim
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg

waves = osim.Waves(height=3.0, amplitude=0.6, windSpeed=7.0)
hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=water, surface_fn=waves), robot)
```

The rendered surface and the surface the hull floats in are the same function.
`Vehicle.apply` advances the waves; without a `Vehicle`, call `waves.step(dt)`.
`surface_fn` accepts any callable from shape positions to surface heights. Waves
move the free surface only; they add no orbital velocity.

## Sensor topics

```python
ros = osim.ros_publishers("isaacswim", camera=True, sonar=True, baro=True)
sensors = {
    "sonar": osim.Sonar(base_prim, ros=ros),
    "baro": osim.Barometer(base_prim, water, ros=ros),
    "camera": osim.Camera(base_prim, ros=ros),
}
```

OceanSim's DVL uses its own message type, built from OceanSim's repository;
`ros_publishers` leaves it out.

## Vehicle topics

```python
from isaacswim.ros2 import CmdVel, publish_vehicle

publish_vehicle("/World/rov", "/World/rov/base_link")
cmd = CmdVel("/cmd_vel")
linear, angular = cmd.read()
```

`publish_vehicle` builds one OmniGraph from Isaac Sim's stock clock, transform,
joint-state and odometry nodes, stamped with simulation time. What a twist means
for thrust is the controller's choice.

## Launching with ROS 2

```python
prepare_launch(args, oceansim=True, ros2=True)
app = AppLauncher(args).app
```

Without a sourced ROS 2 installation, `ros2=True` points the bridge at Isaac
Sim's bundled ROS 2 libraries. On Linux it prints the library path to export
before launch. On Isaac Lab 3, call `isaacswim.ros2.update()` after each control
step.

## Known issues

- **OceanSim asset warning.** OceanSim's example module reports a missing asset
  path unless its asset pack is registered. The sensors and waves still load.
- **h5py with ROS 2 on Windows.** The bridge loads its own HDF5 library, so h5py
  fails in that process. Record demonstrations without `ros2=True`.
- **Windows loader error 448.** Launch with a clean `PATH` containing only the
  Isaac Lab environment and the system directories.
- **`LLVM ERROR: out of memory` soon after launch.** Another ROS 2 system on the
  same network and DDS domain can cause it. Isolate with `ROS_LOCALHOST_ONLY=1`
  and an unused `ROS_DOMAIN_ID`.

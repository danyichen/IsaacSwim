# IsaacSwim

**Underwater robotics add-on for NVIDIA Isaac Sim and Isaac Lab.** IsaacSwim adds MuJoCo's hydrodynamics,
plus buoyancy, currents, thrusters and marine sensors to any articulation or rigid body. Made for
ROVs, AUVs, underwater manipulation, and swimming robots to run in the Isaac ecosystem.

<video class="hero-video" src="media/course_tour_8x.mp4" poster="media/course_tour_poster.jpg" controls muted loop playsinline></video>

*A demo mission made with IsaacSwim's task and scene APIs, at 8× speed.
[Watch at normal speed on YouTube](https://youtu.be/3lMIQrnFJFI).*

Start with [installation](installation.md) and your [first simulation](first_simulation.md), or
[build your own task](tasks.md).

## Features

| | |
|---|---|
| **Water forces** | Drag, lift and added mass from MuJoCo's ellipsoid fluid model, in torch and NVIDIA Warp |
| **Buoyancy** | Displaced volume and centre of buoyancy from meshes, clipped at a flat or moving surface |
| **Any robot** | Fluid shapes read from USD, URDF, MuJoCo models, mass properties or written by hand |
| **Currents** | Uniform, gusting, jet, pipe and boundary-layer flow, ocean-model NetCDF and CFD point clouds |
| **Actuators** | Thrust allocation, propeller and motor models with T200 presets, rate-limited joint servos |
| **Sensors** | Batched depth, DVL, altimeter, IMU and odometry models, plus OceanSim cameras, imaging sonar, DVL and barometer |
| **Integration** | Isaac Lab articulations, batched environments, ROS 2 topics and OceanSim waves |
| **Tasks and scenes** | Your own tasks and USD scenes, scripted experts, parallel evaluation and HDF5 demonstrations |

## Benchmarks

IsaacSwim, Stonefish 1.5 and HoloOcean 2.3 on one RTX 4070 / i7-13700F Linux machine. Accuracy uses native simulator rollouts against recorded BlueROV2 Heavy motion. See the
[benchmark methods](benchmarks.md).

![Physics step time, startup time and camera real-time factor in batched simulation](media/bench_speed.svg#only-light)
![Physics step time, startup time and camera real-time factor in batched simulation](media/bench_speed_dark.svg#only-dark)

![Native prediction error on the Marinarium BlueROV2 datasets](media/bench_accuracy.svg#only-light)
![Native prediction error on the Marinarium BlueROV2 datasets](media/bench_accuracy_dark.svg#only-dark)

## Example

```python
from isaacswim import Medium
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg

hydro = Hydrodynamics(HydrodynamicsCfg(auto_shapes=True, medium=Medium.sea_water()), robot)

while simulation_app.is_running():
    hydro.apply()
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
```

Using IsaacSwim in research? See [how to cite it](citing.md).

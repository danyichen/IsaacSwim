# IsaacSwim

Underwater robotics for NVIDIA Isaac Sim and Isaac Lab:
[github.com/danyichen/IsaacSwim](https://github.com/danyichen/IsaacSwim).

- Hydrodynamics matching MuJoCo's ellipsoid fluid model, with mesh buoyancy and a free surface.
- Fluid shapes read from a robot's geometry, as MuJoCo does, or from mass properties, URDF or MuJoCo models.
- Currents from analytic fields, ocean models or CFD.
- Thrusters, propellers, motors, joint servos and a `Vehicle` that combines them with sensors.
- Navigation sensors, OceanSim sensor adapters and underwater cameras.
- `isaacswim.assets`: a BlueROV2 Heavy with a Reach Alpha 5 arm and Robotiq gripper,
  built from downloaded sources that keep their own licences.

Custom tasks and scenes, scripted experts, parallel simulation and demonstrations are in
the separate `isaacswim-bench` package.

The physics imports without Isaac Sim. See the repository for documentation.

# Changelog

## 0.1.0

Initial research beta for Isaac Sim 5.1 and Isaac Lab 2.3 on Linux and Windows,
with a research preview for Isaac Sim 6.2 and Isaac Lab 3.0.

- Hydrodynamics with drag, lift, added mass and mesh buoyancy, in torch and
  NVIDIA Warp, checked against MuJoCo.
- Fluid shapes read from a robot's geometry per geom, as MuJoCo does, or from
  mass properties, URDF or MuJoCo models.
- Uniform, gusting, jet, pipe, boundary-layer and gridded currents.
- Thruster allocation, propeller and motor models, joint servos and navigation sensors.
- BlueROV2 Heavy asset builder with a Reach Alpha 5 arm and Robotiq 2F-85
  gripper. Every asset is downloaded from a pinned source with its licence; the
  Alpha 5 is for simulation only.
- The separate `isaacswim-bench` package: a framework for custom tasks and scenes,
  with scripted experts, parallel simulation, HDF5 demonstrations and video, plus
  six example tasks.
- Demos: a reef service mission and RSL-RL training.
- OceanSim camera, sonar, DVL, barometer and wave adapters, with the sonar
  displayed as an imaging-sonar fan; ROS 2 integration.

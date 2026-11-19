# Vehicles

`Vehicle` binds an Isaac Lab articulation to hydrodynamics, actuators and
sensors. A controller is any callable that maps observations to actuator
setpoints. Everything is batched over environments.

```python
from isaacswim import IMU, DopplerVelocityLog, Odometry, PressureDepth
from isaacswim.actuators import JointServos, ThrusterArray, Thrusters
from isaacswim.isaaclab import Vehicle

vehicle = Vehicle(
    robot,
    hydro=hydro,
    seabed_z=-4.0,
    actuators=[Thrusters(ThrusterArray("thrusters.json"), body="base_link"),
               JointServos(["arm_.*"], rate=0.5, name="arm")],
    sensors={"depth": PressureDepth(), "dvl": DopplerVelocityLog(),
             "imu": IMU(), "odom": Odometry()},
)

def controller(obs, dt):
    return {"thrusters": body_wrench(obs)}

for step in range(steps):
    if step % decimation == 0:
        vehicle.control(controller, dt * decimation)
    vehicle.apply(dt)
    sim.step()
    vehicle.update(dt)
```

`Vehicle.apply` sums every actuator's wrench with the hydrodynamics into one
write per step, sets joint targets and advances the current field.
`Vehicle.reset(env_ids)` resets actuators, added-mass history and, if given, the controller.

To add your own force, pass it to `vehicle.apply(dt, add=(F, T))` or
`hydro.apply(add=(F, T))`. A second direct wrench write would replace the first.

## Actuators and sensors

[Actuators](actuators.md) covers thrusters, propellers, motors and joint servos, parameter by
parameter. [Navigation sensors](sensors.md) covers depth, DVL, altimeter, IMU and odometry, and
[cameras and sonar](sensing.md) covers the rendered ones.

## Trim

`isaacswim.trim.solve_trim` adds the buoyancy volumes a vehicle needs to be
neutral and positions the movable one so it floats level, as an operator trims
on deck. It reports the resulting separation between centre of buoyancy and
centre of mass.

## Recording frames

`isaacswim.isaaclab.recorder` returns finished frames from the asynchronous RTX
renderer. Cameras are covered in [sensing](sensing.md).

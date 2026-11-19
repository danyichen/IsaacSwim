# Actuators

Actuators go in `Vehicle(actuators=[...])` and receive setpoints from your controller under
their `name`. Their wrenches are summed with hydrodynamics into one write per step.

## Thrusters

`Thrusters(array, ...)` turns a body wrench, or individual thrust commands, into forces at each
thruster.

| Parameter | Default | Meaning |
|---|---|---|
| `array` | required | A `ThrusterArray` with thruster positions and axes |
| `body` | `"base_link"` | Body the thrusters act on |
| `name` | `"thrusters"` | Setpoint key |
| `input` | `"wrench"` | `"wrench"` (6 values) or `"thrust"` (one per thruster) |
| `allocation` | `"nullspace"` | `"nullspace"` respects asymmetric limits; `"pinv"` is the plain pseudo-inverse |
| `time_constant` | 0.0 s | First-order thrust lag |
| `propeller` | `None` | `Propeller` model; thrust then depends on inflow |
| `motor` | `None` | `Motor` model, driven by duty cycle; needs a propeller with torque coefficients |
| `rho` | 1025.0 kg/m³ | Water density used by the propeller |

`thrusters.thrust` holds the applied thrusts and `thrusters.power` the electrical power when a
motor is present.

### Thruster layout

```python
from isaacswim.actuators import ThrusterArray

ThrusterArray("thrusters.json")
ThrusterArray.from_geometry(positions, axes, names=names)
```

`fwd` and `rev` set the thrust limits, 51.48 N and 40.21 N by default. `rank()`, `cond()` and
`max_wrench_axis(k)` show what the layout can do; `wrench_of(thrust)` maps thrusts to a wrench.

### Propeller

Thrust follows rotation rate and falls linearly with advance ratio. `kt_curve=(J, K_T)` uses
measured forward-thrust data instead.

| Parameter | Meaning |
|---|---|
| `diameter` | Propeller diameter, m |
| `kt0_fwd`, `kt0_rev` | Bollard thrust coefficients |
| `j0_fwd`, `j0_rev` | Advance ratio at zero thrust |
| `max_rps_fwd`, `max_rps_rev` | Speed limits, rev/s |
| `wake_fraction` | Inflow reduction |
| `kq0_fwd`, `kq0_rev` | Torque coefficients, needed with a motor |

### Motor

`Motor(ke, resistance, inertia, voltage)` is a DC motor driven by duty cycle.

```python
from isaacswim.actuators import Motor, Propeller, ThrusterArray, Thrusters

thrusters = Thrusters(
    ThrusterArray("thrusters.json"),
    propeller=Propeller.t200(voltage=16.0),
    motor=Motor.t200(voltage=16.0),
    time_constant=0.05,
)
```

`Propeller.t200` and `Motor.t200` are presets for Blue Robotics' T200, with propeller data
covering 10 V to 20 V.

## Joint servos

`JointServos(joints, rate=None, name="arm")` walks joint position targets at a rated speed, so
an arm is never commanded faster than it can move. `joints` are name patterns; `rate` is in
rad/s. `at_setpoint()` reports arrival.

```python
from isaacswim.actuators import JointServos

arm = JointServos(["arm_.*"], rate=0.5, name="arm")
```

## Your own forces

Pass extra wrenches to `vehicle.apply(dt, add=(F, T))` or `hydro.apply(add=(F, T))`. A separate
direct wrench write would replace the sum.

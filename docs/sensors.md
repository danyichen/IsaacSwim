# Navigation sensors

Sensors go in `Vehicle(sensors={...})`; `vehicle.observe()` returns their batched outputs under
the same keys. These are IsaacSwim's own batched models and need no renderer. OceanSim's
[camera, sonar, DVL and barometer](sensing.md#oceansim-sensors) use the same dictionary.

| Sensor | Output | Size |
|---|---|---|
| `PressureDepth` | Depth below the surface, m | 1 |
| `DopplerVelocityLog` | Body velocity, m/s, and a validity mask | 3 + mask |
| `Altimeter` | Height above the seabed, m, and a validity mask | 1 + mask |
| `IMU` | Attitude quaternion and body rates | 4 + 3 |
| `Odometry` | Position, quaternion, body linear and angular velocity | 3 + 4 + 3 + 3 |

Invalid DVL and altimeter readings are zero, so controllers should use the mask. The DVL and
altimeter need the vehicle's `seabed_z`.

## PressureDepth

| Parameter | Default | Meaning |
|---|---|---|
| `surface_height` | 0.0 m | Water surface |
| `density` | 1025.0 kg/m³ | True water density |
| `assumed_density` | `None` | Calibration density; a mismatch scales the reading |
| `sigma` | 0.01 m | Gaussian noise |
| `resolution` | 0.0 m | Quantisation step |
| `up` | (0, 0, 1) | Up direction |

## DopplerVelocityLog

| Parameter | Default | Meaning |
|---|---|---|
| `max_altitude` | 30.0 m | Bottom lock is lost above this |
| `sigma` | 0.002 m/s | Gaussian noise |
| `scale_error` | 0.002 | Fractional scale error |
| `water_track` | `False` | Measure velocity through the water instead of over ground |

## Altimeter

| Parameter | Default | Meaning |
|---|---|---|
| `max_range` | 50.0 m | No return beyond this |
| `min_range` | 0.3 m | Blanking distance |
| `sigma` | 0.01 m | Gaussian noise |

## IMU

| Parameter | Default | Meaning |
|---|---|---|
| `attitude_sigma` | 0.0 rad | Attitude noise |
| `gyro_sigma` | 0.0 rad/s | Rate noise |
| `gyro_bias` | (0, 0, 0) rad/s | Constant rate bias |

## Odometry

Ground truth for baselines and debugging, with optional position noise `pos_sigma` (m).

## Example

```python
from isaacswim import IMU, Altimeter, DopplerVelocityLog, PressureDepth

sensors = {
    "depth": PressureDepth(sigma=0.02),
    "dvl": DopplerVelocityLog(max_altitude=50.0),
    "altitude": Altimeter(),
    "imu": IMU(gyro_sigma=0.002),
}
```

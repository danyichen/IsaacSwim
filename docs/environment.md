# Water and currents

## Water properties

`Medium` describes the fluid:

| Field | Default | Meaning |
|---|---|---|
| `density` | 997.0 kg/m³ | Fluid density |
| `viscosity` | 1.0016e-3 Pa·s | Dynamic viscosity |
| `gravity` | (0, 0, -9.81) m/s² | Gravity vector |
| `current` | (0, 0, 0) m/s | Uniform current, when no field is given |
| `surface_height` | `None` | Free-surface height; `None` means fully submerged |

```python
from isaacswim import Medium

Medium.sea_water()              # 1025 kg/m³, 1.22e-3 Pa·s
Medium.fresh_water(20.0)        # 998.2 kg/m³, 1.002e-3 Pa·s
Medium.mujoco_default()         # 1000 kg/m³, 1e-3 Pa·s
Medium(density=1025.0, surface_height=0.0)
```

## Free surface and waves

Set `surface_height` for a flat surface. `HydrodynamicsCfg(surface_fn=...)` replaces it with
any function from shape positions to surface heights, such as
[OceanSim waves](ecosystem.md#waves).

## Currents

Pass a field as `HydrodynamicsCfg(current_fn=field)`. A field maps positions to velocities
and advances with time; `Vehicle.apply` steps it for you.

| Field | Models | Main parameters |
|---|---|---|
| `Uniform` | Constant flow, optionally slowed near the seabed | `velocity`, `seabed_z`, `layer` |
| `GustyCurrent` | A mean flow with correlated random gusts | `mean`, `sigma`, `tau`, `seabed_z`, `layer`, `seed` |
| `Jet` | A round turbulent jet from an outlet | `point`, `direction`, `radius`, `outlet_velocity`, `decay`, `spread` |
| `Pipe` | Flow through a tapered pipe | `p1`, `p2`, `r1`, `r2`, `v_in`, `profile`, `n`, `nu` |
| `Gridded` | Sampled data, optionally time-varying | `velocity`, `x`, `y`, `z`, `times`, `outside`, `loop` |
| `BoundaryLayer` | Power-law slowing of any field near the seabed | `field`, `seabed_z`, `thickness`, `exponent` |
| `Composite` | The sum of several fields | `*fields` |

```python
from isaacswim.flow import BoundaryLayer, Composite, Gridded, GustyCurrent, Jet, Uniform

Uniform((0.2, 0.0, 0.0), seabed_z=-4.0, layer=1.5)
GustyCurrent(mean=(0.15, 0.0, 0.0), sigma=(0.05, 0.05, 0.0), tau=2.5, seabed_z=-4.0)
Jet((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), radius=0.05, outlet_velocity=1.0)
Composite(Uniform((0.1, 0.0, 0.0)), Jet((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 0.05, 1.0))
```

### Ocean and CFD data

```python
from isaacswim.flow import Gridded

Gridded.from_netcdf("site.nc", origin_lonlat=(lon, lat), loop=True)   # needs the netcdf extra
Gridded.from_scattered(points, velocity, spacing=0.05)                # needs the cfd extra
```

## Water appearance

Cameras use [OceanSim's image model](sensing.md#water): veiling colour, attenuation and
backscatter, including presets for Jerlov water types.

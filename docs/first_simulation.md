# First simulation

This example drops an aluminium ball into water and checks that it reaches its theoretical
terminal speed of 1.4911 m/s.

## Without Isaac Sim

```bash
python scripts/tutorials/sphere_terminal_velocity.py
```

## In Isaac Sim

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tutorials/first_swimmer.py
```

[`first_swimmer.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/tutorials/first_swimmer.py)
spawns a rigid ball, then adds water forces each physics step:

```python
from isaacswim import FluidShape, Medium
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg

water = Medium(density=1000.0, viscosity=1.0e-3)
shape = FluidShape.sphere(ball.body_names[0], 0.05)
hydro = Hydrodynamics(HydrodynamicsCfg(shapes=[shape], medium=water), ball)

for _ in range(steps):
    hydro.apply()
    ball.write_data_to_sim()
    sim.step(render=False)
    ball.update(dt)
```

Create `Hydrodynamics` after `sim.reset()`. Next, add water to [your robot](hydrodynamics.md) or
[build a vehicle](vehicles.md).

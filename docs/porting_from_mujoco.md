# Porting from MuJoCo

IsaacSwim implements MuJoCo's ellipsoid fluid model, so an underwater MuJoCo
model behaves the same in Isaac Sim. Frames, merged links, joints and buoyancy
still need checking.

## Import

Convert the MJCF with Isaac Lab's importer and let `auto_shapes` read one
ellipsoid per geom, as MuJoCo does:

```python
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

enable_extension("isaacsim.asset.importer.mjcf")
usd = MjcfConverter(MjcfConverterCfg(asset_path="swimmer.xml", usd_dir="usd")).usd_path
robot = Articulation(ArticulationCfg(
    prim_path="/World/swimmer",
    spawn=sim_utils.UsdFileCfg(usd_path=usd),
    articulation_root_prim_path="/torso/torso",
    actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0,
                                             damping=0.0)},
))
sim.reset()
hydro = Hydrodynamics(HydrodynamicsCfg(auto_shapes=True, medium=Medium(density=1000.0)),
                      robot)
```

The importer adds an empty `worldBody` articulation root, so name the real root
body with `articulation_root_prim_path`. `auto_shapes` uses MuJoCo's default
fluid coefficients; a model that sets `fluidcoef` should transfer its shapes
instead, as below.

`scripts/tools/check_mujoco_ingestion.py` imports a multi-link MJCF swimmer,
runs it in both simulators under the same forces and compares shapes and
trajectories. Pass `--mjcf` to run it on your own model.

## Transfer

To keep the compiled shapes and coefficients exactly, on a machine with MuJoCo:

```bash
python -m isaacswim.mjcf model/scene.xml fluid.json
python scripts/tutorials/port_from_mujoco.py verify model/scene.xml
```

In Isaac Sim:

```python
from isaacswim.mjcf import load_json

shapes, medium = load_json("fluid.json")
```

`shapes_from_mjmodel` reads the compiled model, so defaults, mesh-derived sizes
and MuJoCo's own fluid coefficients are used as compiled.

## What to check

**Body names.** Map MuJoCo bodies to USD links with
`shapes_from_mjcf("scene.xml", rename={"base_link": "hull"})`. A rename is only
correct when the two frames coincide; otherwise every shape is misplaced.

**Merged links.** When several MuJoCo bodies become one link, compose each geom's
offset through the intervening body transforms and keep each ellipsoid as its
own shape.

**Ball joints.** Isaac Lab drives are single-axis. Two revolute joints about one
point keep both bending axes; give each the full ball stiffness, and give the
intermediate link a real mass.

**Gravity compensation.** `gravcomp="1"` is neutral buoyancy. Its equivalent is
zero scene gravity with `HydrodynamicsCfg(buoyancy=False)`. For a vehicle that is
not neutral, keep gravity, set each shape's displaced volume and let IsaacSwim
supply buoyancy.

**Velocity and force frames.** `Hydrodynamics` uses centre-of-mass velocities
and applies moments about the centre of mass. Add other wrenches through
`apply(add=(F, T))` in the same convention.

## Validating a port

1. `port_from_mujoco.py verify`: the fluid model alone.
2. Static pose under the same joint torques, fluid off: kinematics and springs.
3. Free deceleration from the same initial velocity: fluid with the real inertia.
4. The same open-loop command in both simulators: actuation.

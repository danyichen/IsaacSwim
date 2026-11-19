# Hydrodynamics

`Hydrodynamics` computes the water's force and torque on every body of an Isaac
Lab articulation or rigid object and writes them once per physics step.

## Fluid shapes

Each force acts on a `FluidShape`: an ellipsoid attached to a body, standing in
for the water that body moves. A body can carry several shapes, for example one
per segment of a slender arm. A shape may also carry a mesh, which then sets its
displaced volume and centre of buoyancy.

The [physics model](physics.md) gives the equations for drag, lift, added mass and buoyancy.

## Describing a robot

Let IsaacSwim read the shapes from the robot, or describe them yourself.

**Automatically.** One ellipsoid per geometry prim, sized the way MuJoCo sizes a
geom with `fluidshape="ellipsoid"`: spheres, capsules, cylinders and boxes by
their dimensions, meshes by their bounding box in the principal-inertia frame.
Collision geometry is used when a body has any, visual geometry otherwise.
Meshes also set the displaced volume and centre of buoyancy:

```python
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg

hydro = Hydrodynamics(HydrodynamicsCfg(auto_shapes=True), robot)
```

`auto_body_pattern` selects bodies. `auto_shapes="inertia"` instead fits one
ellipsoid per body to its mass and inertia, with `body_density` setting each
body's displaced volume from its mass.

**From geometry, explicitly.** The same readers, for a stage, a USD file or a
URDF:

```python
from isaacswim.urdf import shapes_from_urdf
from isaacswim.usd import shapes_from_articulation_usd, shapes_from_usd

shapes = shapes_from_articulation_usd(robot)
shapes = shapes_from_usd("robot.usd")
shapes = shapes_from_urdf("robot.urdf")
```

**From MuJoCo.** See [porting from MuJoCo](porting_from_mujoco.md):

```python
from isaacswim.mjcf import shapes_from_mjcf

shapes, medium = shapes_from_mjcf("scene.xml")
```

**By hand.**

```python
from isaacswim import FluidShape

FluidShape(body="hull", semiaxes=(0.3, 0.2, 0.15), pos=(0.0, 0.0, 0.05))
FluidShape.capsule("arm_3", radius=0.02, half_length=0.08, axis="z")
FluidShape.box("hull", (0.3, 0.2, 0.15), mesh=True)
FluidShape.from_mesh("hull", points, faces)
```

Then pass them in the config:

```python
hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=medium), robot)
```

## Coefficients

Drag and lift coefficients default to MuJoCo's (`DEFAULT_COEF`). Override them
per shape with `coef`. Tow-tank, CFD or drop-test data can set them for a
specific vehicle; randomising them suits learned controllers.

## Buoyancy and the free surface

`Medium` holds density, viscosity, gravity and the surface height:

```python
from isaacswim import Medium

Medium.sea_water()
Medium.fresh_water()
Medium(density=1025.0, surface_height=0.0)
```

A shape with a mesh displaces the volume the mesh encloses, with its centroid as
the centre of buoyancy. At the surface the mesh is clipped against the water
plane, so a surfacing vehicle gets the right righting moment at any attitude.
Shapes without a mesh use the ellipsoid's closed form. `volume` overrides the
magnitude while the wetted fraction still follows the geometry, which suits
flooded frames. `surface_fn` replaces the flat surface with a moving one, such
as [OceanSim waves](ecosystem.md#waves).

For a neutrally buoyant model with gravity compensated elsewhere, set
`buoyancy=False`.

## Added mass

MuJoCo omits the acceleration term of added mass. `HydrodynamicsCfg(added_mass=True)`
applies it without the unstable feedback of a finite difference: it recovers the
forces PhysX added in the previous step and solves for the correction. Call
`apply()` every physics step and `reset(env_ids)` after teleporting.

## Currents

Set `current_fn` to any [velocity field](environment.md#currents):

```python
from isaacswim.flow import Uniform

hydro = Hydrodynamics(
    HydrodynamicsCfg(auto_shapes=True, current_fn=Uniform((0.2, 0.0, 0.0))), robot
)
```

## Backends

Float32 runs as one NVIDIA Warp kernel over environments and shapes; float64
and the fallback use torch, which is the reference.

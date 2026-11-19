# Physics model

IsaacSwim adds water forces to Isaac Sim. PhysX handles rigid-body dynamics, joints and contact;
IsaacSwim computes one force and torque per body and applies them each physics step.

## Conventions

Units are SI and angles are radians unless a name says otherwise. The world frame is Z-up with
gravity along -Z. Quaternions are `(w, x, y, z)`. Tensors are batched as
`(environments, bodies, ...)`.

## Hydrodynamics

Forces act on *fluid shapes*: ellipsoids attached to bodies. Each shape sees the water velocity
at its centre, including any current. The model follows
[MuJoCo's ellipsoid fluid model](https://mujoco.readthedocs.io/en/stable/computation/fluid.html),
with coefficients defaulting to MuJoCo's values.

### Drag

Linear drag combines viscous drag at low speed with quadratic blunt and slender drag based on the
area the shape presents to the flow. Angular drag damps rotation in the same way.

### Lift

Kutta lift comes from the angle of attack and Magnus lift from spin.

### Added mass

Added mass comes from the ellipsoid's Lamb coefficients. The velocity-dependent terms are always
included. `added_mass=True` also applies the acceleration term, which MuJoCo omits, using the
forces PhysX applied in the previous step to stay stable.

### Buoyancy

Buoyancy acts at the centre of displaced volume. Shapes with a mesh use its enclosed volume and
are clipped against the water surface; shapes without one use the ellipsoid's closed form. Drag
and lift scale with the submerged fraction.

## Limitations

Bodies do not disturb each other's water, and wakes, vortex shedding and wave orbital velocity
are not modelled. Drag is based on ellipsoids rather than computational fluid dynamics. The
acceleration added-mass term assumes full submersion, and mesh buoyancy assumes a closed surface.

## Backends

Float32 runs as an NVIDIA Warp kernel. Float64 and the fallback use PyTorch, the reference
implementation.

# BlueROV2 Heavy

The example vehicle: a stock BlueROV2 Heavy with a Reach Alpha 5 arm and a
Robotiq 2F-85 gripper, built as one free-floating articulation.

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tools/build_bluerov.py
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tools/check_bluerov.py
```

The builder downloads pinned files, checks their hashes, and writes the URDF,
USD, meshes and thruster layout to `ISAACSWIM_CACHE`. No meshes are included in
this repository or its wheels. `check_bluerov.py` checks the structure, mass,
free fall and joint tracking in air, and prints the joint order actions index
into. The URDF step alone, `python -m isaacswim.assets.bluerov.build_urdf`, needs
no Isaac Sim.

## Sources

| Part | Source | Licence |
|---|---|---|
| BlueROV2 Heavy: hull, camera tilt, thrusters | [edxmorgan/uvms-simulator](https://github.com/edxmorgan/uvms-simulator) `98e8ba1e` | AGPL-3.0-or-later |
| Reach Alpha 5 | [Robotic-Decision-Making-Lab/alpha](https://github.com/Robotic-Decision-Making-Lab/alpha) `5e1a6648` | Reach Robotics' licence |
| Robotiq 2F-85 | [robotiq/isaacsim_assets](https://github.com/robotiq/isaacsim_assets) | BSD-3-Clause, with CC BY 4.0 portions from NVIDIA |

The build fails if Reach's licence file is missing or altered, and copies
every licence and notice next to the generated files with an
`ASSET_SOURCES.json` provenance record. The Alpha 5 is used for simulation only,
under the conditions in [Licensing](licensing.md#reach-alpha-5).

Local checkouts at the pinned commits can be used instead of downloads:

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tools/build_bluerov.py \
    --upstream-dir /path/to/uvms-simulator --reach-dir /path/to/alpha
```

`--no-robotiq` keeps the Alpha's standard jaws.

## What the build changes

The descriptions target ROS 2 and Gazebo. For PhysX, the builder:

- gives massless links inertia from their collision geometry;
- drops the ground-projection frame;
- mounts the arm on the vehicle's manipulator mount, with nothing welded to the world;
- replaces the jaws' mimic chain with directly driven fingers;
- welds the propeller joints, since thrust is an external wrench;
- makes mesh paths relative so the output folder can move;
- writes `thrusters.json` with each thruster's pose and the allocation matrix;
- validates the kinematic tree before USD conversion.

`build_urdf` keeps the upstream behaviour with `--jaws mimic` and `--thrusters keep`.

Positive-thrust direction is an ESC wiring convention: check that commanding up
moves the vehicle up before trusting an allocation.

## In the water

```python
from isaacswim.assets.bluerov import (ARM_HOME, DEFAULT_USD, build_hydro,
                                      gripper_shapes, robot_cfg)

robot = Articulation(robot_cfg(DEFAULT_USD, pos=(0, 0, -1), quat=(1, 0, 0, 0),
                               joint_pos=ARM_HOME))
sim.reset()
hydro, info = build_hydro(robot, DEFAULT_USD.with_suffix(".urdf"),
                          extra_geom=gripper_shapes(DEFAULT_USD.parent))
```

`build_hydro` takes buoyancy from the URDF meshes, splits the frame mesh into
its largest closed parts for drag, and trims the vehicle neutral and level with
foam the CAD does not contain. `info` reports the trim. Reference controllers,
`StationKeeper`, `BatchStationKeeper` and `AlphaIK`, are in
`isaacswim.assets.bluerov.controllers`.

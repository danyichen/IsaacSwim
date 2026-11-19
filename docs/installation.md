# Installation

IsaacSwim is two packages: the library, `source/isaacswim`, and the task framework,
`source/isaacswim-bench`:

| Module | |
|---|---|
| `isaacswim` | Hydrodynamics, buoyancy, currents, actuators, sensors and the Isaac Lab binding |
| `isaacswim.assets` | The BlueROV2 Heavy asset builder and its controllers |
| `isaacswim_bench` | Your own tasks and scenes, experts, evaluation and demonstrations (`isaacswim-bench`) |

The task framework is optional.

## Isaac Sim and Isaac Lab

Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
for your Isaac Sim version and set `ISAACLAB_PATH` to the checkout. Accept
NVIDIA's licence for that interpreter before running unattended. Install with
`--no-deps` so the simulator keeps its bundled torch.

Linux:

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-deps -e source/isaacswim -e source/isaacswim-bench
"${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install xacro trimesh pycollada h5py \
    imageio imageio-ffmpeg pillow opencv-python-headless
```

Windows PowerShell:

```powershell
& "$env:ISAACLAB_PATH/isaaclab.bat" -p -m pip install --no-deps -e source/isaacswim -e source/isaacswim-bench
& "$env:ISAACLAB_PATH/isaaclab.bat" -p -m pip install xacro trimesh pycollada h5py imageio imageio-ffmpeg pillow opencv-python-headless
```

The remaining examples use the Linux launcher. On Windows, replace
`"${ISAACLAB_PATH}/isaaclab.sh" -p` with `& "$env:ISAACLAB_PATH/isaaclab.bat" -p`.

## Isaac Sim 6 and Isaac Lab 3

Isaac Sim 5.1 with Isaac Lab 2.3 is the supported configuration and the one the
benchmarks use. Isaac Sim 6.2 with Isaac Lab 3.0 is a **research preview**: a
compatibility layer handles Isaac Lab 3's API changes, so the same code runs on both, but
it is less tested and our workloads step slower on it.

To try it, install Isaac Sim 6.2 and Isaac Lab 3.0 (Python 3.12), point `ISAACLAB_PATH`
at that checkout and install IsaacSwim with the same commands as above. Then build the
vehicle for it:

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tools/build_bluerov.py
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tutorials/first_swimmer.py
```

What changes on Isaac Lab 3:

- There is no `--headless` flag. Scripts run headless unless you pass `--visualizer kit`.
- Isaac Lab 3 returns Warp arrays and (x, y, z, w) quaternions. IsaacSwim converts
  internally and always uses torch tensors and (w, x, y, z), so convert only when you read
  Isaac Lab's robot data yourself.
- The vehicle builds into `ISAACSWIM_CACHE/isaacsim6`, so both versions can share one cache.

## Launching

Call `prepare_launch(args)` before `AppLauncher(args)` in your own scripts. On
Windows it selects Direct3D 12 rendering, which avoids a Vulkan crash seen with
some drivers. It also enables OceanSim and ROS 2 when asked:

```python
from isaaclab.app import AppLauncher
from isaacswim import prepare_launch

prepare_launch(args, oceansim=True, ros2=True)
app = AppLauncher(args).app
```

## The example vehicle

The BlueROV2 Heavy is built from downloaded descriptions into
`ISAACSWIM_CACHE` (default `~/.cache/isaacswim`):

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/tools/build_bluerov.py
```

See [BlueROV2 Heavy](bluerov.md). The core library does not need it.

## Optional components

- **OceanSim** is an Isaac Sim extension, installed separately; see
  [its instructions](https://github.com/umfieldrobotics/OceanSim) and
  [OceanSim and ROS 2](ecosystem.md).
- **ROS 2** uses Isaac Sim's bundled bridge; see [OceanSim and ROS 2](ecosystem.md#launching-with-ros-2).
- **Training** needs Isaac Lab's RSL-RL dependencies.
- **Demonstrations** need `h5py`; videos need `imageio` and `imageio-ffmpeg`.

## Next

Run your [first simulation](first_simulation.md).

## Without Isaac Sim

The fluid model, buoyancy, currents, actuators, sensors and task logic run in a
regular Python environment:

```bash
python -m pip install -e "source/isaacswim[all]" -e "source/isaacswim-bench[all]"
```

Extras: `mujoco` (model import and parity tests), `warp` (GPU kernels), `netcdf`
(ocean data), `cfd` (scattered CFD fields), `build` (asset builder), `all` (every extra).
`isaacswim-bench` adds `data` (HDF5 demonstrations), `scenery` (image tools used by the sonar view and demos),
`video` and `all`.

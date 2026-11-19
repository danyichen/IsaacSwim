# Tasks and scenes

`isaacswim_bench` is a small framework for building your own underwater tasks. You describe
what the world contains and what success means; it gives you the simulated robot, sensors, a
kinematic test plant, parallel evaluation, demonstration recording and video.

## Build a task

A task is a Python class. [`custom_task.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/tutorials/custom_task.py)
is a complete example. Run it with or without Isaac Sim:

```bash
python -m isaacswim_bench.run --task scripts.tutorials.custom_task:BuoyHover \
    --plant kinematic --episodes 5 --out runs/buoy
```

Any importable `module:Class` works as `--task`. A task implements:

- `sample(rng)`: random parameters for an episode.
- `scene(params)`: a `SceneSpec` with objects, spawn pose and current.
- `evaluate(world, params, mem, dt)`: a `StepResult` with reward, success or failure.
- `privileged(world, params)`: ground-truth state for baselines.
- `expert(params, arm)`: an optional scripted policy.

## Build a scene

`Prop` adds boxes, cylinders and spheres as `static`, `dynamic` or `visual` objects. For meshes,
materials, lights or a whole USD level, override `dress`:

```python
from pxr import UsdGeom

def dress(self, stage, *, seed=0, root="/World", origin=(0.0, 0.0, 0.0)):
    path = f"{root}/site"
    site = UsdGeom.Xform.Define(stage, path)
    site.GetPrim().GetReferences().AddReference("/path/to/my_scene.usd")
    site.AddTranslateOp().Set(tuple(float(v) for v in origin))
    return (path,)
```

Override `label_sonar` to make that geometry visible to the imaging sonar. `report_fields` adds
reported values to the action.

## Examples

`station_keep`, `pipeline_inspect`, `target_approach`, `dock`, `sample_collect` and `valve_turn`
are included. [`scripts/demos/reef_tour`](https://github.com/danyichen/IsaacSwim/tree/main/scripts/demos/reef_tour)
is the larger mission shown on the [home page](index.md).

## Observations and actions

Every task uses the [BlueROV2 Heavy](bluerov.md) with the same observations and actions, so one
policy can train across tasks.

| Key | |
|---|---|
| `rgb_front`, `rgb_arm` | Hull and forearm cameras through the underwater image model |
| `sonar` | OceanSim imaging sonar, with `--sonar` |
| `depth`, `altitude`, `imu`, `dvl` | Instrument models with noise; OceanSim's with `--oceansim` |
| `arm_q`, `gripper` | Joint angles and gripper state |
| `proprio` | The proprioceptive keys concatenated |
| `instruction` | The task instruction |
| `priv_task`, `priv_odom` | Privileged ground truth, for baselines only |

Public observations carry no horizontal position. The default station-keeper
flies on the vehicle's own dead-reckoned navigation; `--nav truth` uses ground
truth instead.

The action is surge, sway, heave and yaw rate, four arm joint rates and grip,
each in [-1, 1]. Velocities move a setpoint that the station-keeper holds, so a
zero action holds position. `--action-mode wrench` commands a body wrench
instead.

## Running

```bash
python -m isaacswim_bench.run --task dock --plant kinematic --episodes 20 --out runs/kinematic

"${ISAACLAB_PATH}/isaaclab.sh" -p -m isaacswim_bench.run \
    --task valve_turn --num-envs 8 --episodes 32 --policy expert \
    --views none --out runs/valve
```

The kinematic plant checks task logic without Isaac Sim. Each run writes
`episodes.jsonl` and `summary.json` with success rate and confidence interval.
Use a fresh `--out` for every run.

## Python API

```python
from isaacswim_bench import KinematicEnv, make_task

env = KinematicEnv(make_task("valve_turn"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.expert_action())
```

After launching Isaac Sim, `isaacswim_bench.sim.SimEnv` has the same interface
and `VectorSimEnv` runs environments in parallel with independent resets:

```python
from isaacswim_bench.sim import VectorSimEnv

env = VectorSimEnv("valve_turn", num_envs=8, device="cuda:0", views=())
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.expert_action())
done = (terminated | truncated).nonzero()[0].tolist()
env.reset(seed=[1000 + i for i in done], env_ids=done)
```

[`record_demos.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/tutorials/record_demos.py) and
[`parallel_envs.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/tutorials/parallel_envs.py) are complete examples.

## Demonstrations

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p -m isaacswim_bench.run \
    --task sample_collect --num-envs 4 --episodes 32 --policy expert \
    --views front,arm --image-size 128 128 \
    --record runs/samples.hdf5 --out runs/samples
python scripts/tutorials/inspect_demos.py runs/samples.hdf5
```

Files follow robomimic's HDF5 layout, with the instruction in each demo's
`ep_meta["lang"]` and successful demos in `mask/successful`. Failed episodes are
kept only with `--keep-failed`. Load them with
`isaacswim_bench.dataset.load_demos`.

`--video` writes a chase and wrist video per episode without changing what the
policy sees. `--views front,arm,down --sonar` records all sensor streams, which
[`sensor_video.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/tutorials/sensor_video.py) replays without Isaac Sim.

## Evaluating a policy

`--policy module:factory` loads `factory(env)`, which returns
`policy(obs) -> action` with an optional `reset(instruction)`:

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p -m isaacswim_bench.run \
    --task dock --episodes 10 --seed0 10000 --views none \
    --policy scripts.tutorials.example_policy:load --out runs/policy
```

Keep evaluation seeds separate from demonstration seeds.

## Reinforcement learning

[`scripts/demos/rl`](https://github.com/danyichen/IsaacSwim/tree/main/scripts/demos/rl) shows how
to train on these tasks with Isaac Lab's RSL-RL trainer:

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/demos/rl/train.py \
    --task IsaacSwim-BlueROV-Dock-v0 --num_envs 8 --max_iterations 300
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/demos/rl/play.py \
    --task IsaacSwim-BlueROV-Dock-v0 --checkpoint PATH_TO_MODEL --num_envs 2
```

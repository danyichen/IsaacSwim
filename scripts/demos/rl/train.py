"""Train a task with RSL-RL PPO across parallel environments."""

from __future__ import annotations

import argparse
import time

from rsl_rl.runners import OnPolicyRunner  # isort: skip
from isaaclab.app import AppLauncher  # isort: skip

from isaacswim.launch import prepare_launch  # isort: skip

parser = argparse.ArgumentParser(description="Train an IsaacSwim task with RSL-RL PPO.")
parser.add_argument("--task", default="IsaacSwim-BlueROV-StationKeep-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_root", default="logs/rsl_rl")
parser.add_argument("--run_name", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
prepare_launch(args)
app = AppLauncher(args).app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import rl_envs  # noqa: E402,F401  (registers the tasks)

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
if hasattr(env_cfg, "task_name"):
    agent_cfg.experiment_name = f"isaacswim_{env_cfg.task_name}"
env_cfg.seed = agent_cfg.seed = args.seed
env_cfg.sim.device = agent_cfg.device = args.device
log_dir = os.path.abspath(os.path.join(
    args.log_root, agent_cfg.experiment_name,
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else "")))
env_cfg.log_dir = log_dir

t0 = time.time()
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, rl_envs.runner_cfg(agent_cfg), log_dir=log_dir,
                        device=agent_cfg.device)
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
t1 = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
print(f"setup {t1 - t0:.1f} s, training {time.time() - t1:.1f} s, logs in {log_dir}", flush=True)
if hasattr(env_cfg, "task_name"):
    import sys
    writer = getattr(runner, "writer", None)
    if writer is not None:
        writer.flush()
        writer.close()
    logger = getattr(runner, "logger", None)
    if logger is not None and hasattr(logger, "close"):
        logger.close()
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
env.close()
app.close()

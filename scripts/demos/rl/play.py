"""Run a trained RSL-RL policy in a task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run a saved RSL-RL policy.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path,
                        help="defaults to params/agent.yaml beside the checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10000)
    pre, _ = parser.parse_known_args()
    if not pre.checkpoint.is_file():
        parser.error("--checkpoint must name an existing model file")
    if pre.num_envs < 1 or pre.steps < 1:
        parser.error("environment and step counts must be positive")
    if pre.agent_config and not pre.agent_config.is_file():
        parser.error("--agent-config must name an existing YAML file")
    from isaaclab.app import AppLauncher
    from rsl_rl.runners import OnPolicyRunner

    from isaacswim.launch import prepare_launch

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    app = AppLauncher(prepare_launch(args)).app
    import gymnasium as gym
    import torch
    from isaaclab.utils.io import load_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    import rl_envs

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    saved_config = args.agent_config or args.checkpoint.parent / "params/agent.yaml"
    if saved_config.is_file():
        agent.from_dict(load_yaml(str(saved_config)))
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = agent.device = args.device
    cfg.seed = agent.seed = args.seed
    base = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, rl_envs.runner_cfg(agent),
                                log_dir=None, device=args.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        observations = env.get_observations()
        completed = steps = 0
        reward_sum = 0.0
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("policy produced nonfinite actions")
                observations, rewards, dones, _ = env.step(actions)
                if not torch.isfinite(rewards).all():
                    raise RuntimeError("environment produced nonfinite rewards")
                reward_sum += float(rewards.sum())
                completed += int(dones.sum())
                steps += 1
                if not args.headless:
                    base.unwrapped.sim.render()
        print(json.dumps({"task": args.task, "checkpoint": str(args.checkpoint),
                          "num_envs": args.num_envs, "control_steps": steps,
                          "completed_episodes": completed,
                          "mean_reward_per_transition": reward_sum / max(1, steps * args.num_envs)},
                         indent=2), flush=True)
    finally:
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

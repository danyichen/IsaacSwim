"""Step many copies of a task in parallel, resetting each one as it finishes."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Step parallel task environments.")
    parser.add_argument("--task", default="valve_turn")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    pre, _ = parser.parse_known_args()
    if pre.num_envs < 1 or pre.steps < 1:
        parser.error("environment and step counts must be positive")
    from isaaclab.app import AppLauncher

    from isaacswim.launch import prepare_launch

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    args = parser.parse_args()
    app = AppLauncher(prepare_launch(args)).app
    import numpy as np

    from isaacswim_bench.sim import VectorSimEnv

    env = VectorSimEnv(args.task, num_envs=args.num_envs, device=args.device, views=())
    episodes = 0
    next_seed = args.seed + args.num_envs
    try:
        observations, _ = env.reset(seed=args.seed)
        print("Batched observation shapes:", {
            key: list(value.shape) for key, value in observations.items()
            if hasattr(value, "shape")
        }, flush=True)
        for _ in range(args.steps):
            if not app.is_running():
                break
            actions = env.expert_action()
            _, rewards, terminated, truncated, _ = env.step(actions)
            if not args.headless:
                env.sim.render()
            if not np.isfinite(rewards).all():
                raise RuntimeError("nonfinite rewards")
            ids = np.flatnonzero(terminated | truncated).tolist()
            for i in ids:
                print(json.dumps({"slot": i, **env.envs[i].report()}), flush=True)
            if ids:
                seeds = list(range(next_seed, next_seed + len(ids)))
                env.reset(seed=seeds, env_ids=ids)
                next_seed += len(ids)
                episodes += len(ids)
        print(json.dumps({"task": args.task, "num_envs": args.num_envs,
                          "completed_episodes": episodes}), flush=True)
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

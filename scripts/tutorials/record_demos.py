"""Run a task with its scripted expert and record demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Step a task and record demonstrations.")
    parser.add_argument("--task", default="dock")
    parser.add_argument("--plant", choices=("sim", "kinematic"), default="sim")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--views", default="none", help="none or front,arm")
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    pre, _ = parser.parse_known_args()
    if pre.episodes < 1 or (pre.max_steps is not None and pre.max_steps < 1):
        parser.error("episode and step counts must be positive")
    if pre.record and pre.record.exists():
        parser.error("choose a fresh --record path")
    if pre.plant == "kinematic" and pre.views != "none":
        parser.error("camera views require --plant sim")
    if pre.plant == "sim":
        from isaaclab.app import AppLauncher

        from isaacswim.launch import prepare_launch

        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(device="cpu")
        args = parser.parse_args()
        args.enable_cameras = args.views != "none"
        app = AppLauncher(prepare_launch(args)).app
    else:
        args = parser.parse_args()
        app = None

    from isaacswim_bench import KinematicEnv, make_task, summarize
    from isaacswim_bench.dataset import DemoWriter

    task = make_task(args.task)
    if app is None:
        env = KinematicEnv(task)
    else:
        from isaacswim_bench.sim import SimEnv

        env = SimEnv(task, device=args.device, views=() if args.views == "none"
                     else args.views.split(","), image_size=tuple(args.image_size))
    writer = None
    reports = []
    try:
        if args.record:
            writer = DemoWriter(args.record, keep_failed=args.keep_failed, env_args={
                "env_name": f"isaacswim_bench/{task.name}", "type": "isaacswim_bench",
                "env_kwargs": {"task": task.name, "plant": args.plant, "control_hz": 20,
                               "action_mode": "pilot", "views": args.views,
                               "image_size": list(args.image_size)},
            })
        for episode in range(args.episodes):
            if app is not None and not app.is_running():
                break
            obs, info = env.reset(seed=args.seed + episode)
            print(info["instruction"], flush=True)
            if episode == 0:
                print("Observation shapes:", {key: list(value.shape) for key, value in obs.items()
                                              if hasattr(value, "shape")}, flush=True)
            if writer:
                writer.start(env)
            step = 0
            terminated = False
            while True:
                if app is not None and not app.is_running():
                    break
                action = env.expert_action()
                next_obs, reward, terminated, truncated, info = env.step(action)
                if writer:
                    writer.add(obs, action, reward, terminated)
                obs = next_obs
                if app is not None and not args.headless:
                    env.sim.render()
                step += 1
                if terminated or truncated or (args.max_steps and step >= args.max_steps):
                    break
            report = env.report()
            report["truncated"] = not bool(terminated)
            if writer:
                report["recorded"] = writer.end(report)
            reports.append(report)
            print(json.dumps(report), flush=True)
        print(json.dumps(summarize(reports), indent=2), flush=True)
        if writer:
            print(f"Saved {writer.num_demos} demonstrations to {args.record}", flush=True)
    finally:
        if writer:
            writer.close()
    if app is not None:
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

"""Evaluate policies and record demonstrations: python -m isaacswim_bench.run."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

EVAL_SEED0 = 10000


def _args(argv=None):
    ap = argparse.ArgumentParser(description="Run a task benchmark.")
    ap.add_argument("--task", required=True, help="a task name or module:Class")
    ap.add_argument("--num-envs", type=int, default=1,
                    help="parallel task environments in one physics scene")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0,
                    help=f"first episode seed; evaluation convention is {EVAL_SEED0}")
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--plant", choices=("sim", "kinematic"), default="sim")
    ap.add_argument("--out", required=True, help="reports go here -- not a synced folder")
    ap.add_argument("--record", default=None, help="robomimic-style HDF5 of the episodes")
    ap.add_argument("--keep-failed", action="store_true",
                    help="record failed episodes too (default: successes only)")
    ap.add_argument("--action-mode", choices=("pilot", "wrench"), default="pilot")
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="always the canonical instruction")
    ap.add_argument("--sensor-noise", type=float, default=1.0)
    ap.add_argument("--views", default="front,arm",
                    help="comma-separated camera views, or 'none'")
    ap.add_argument("--image-size", type=int, nargs=2, default=(128, 128),
                    metavar=("H", "W"))
    ap.add_argument("--water", default="clear_ocean",
                    help="camera water: clear_ocean | hazy_ocean | harbour | none")
    ap.add_argument("--sonar", action="store_true", help="mount OceanSim's imaging sonar")
    ap.add_argument("--nav", choices=("dvl", "truth"), default="dvl",
                    help="station-keeper on dead-reckoned DVL navigation, or ground truth")
    ap.add_argument("--oceansim", action="store_true",
                    help="depth and DVL from OceanSim's barometer and raycast DVL")
    ap.add_argument("--video", action="store_true",
                    help="an mp4 per episode: chase view with thrust arrows | wrist view")
    ap.add_argument("--video-height", type=int, default=540, help="video pixel height")
    ap.add_argument("--current-scale", type=float, default=1.0,
                    help="multiply each episode's current (mean and gusts); 1 = benchmark")
    ap.add_argument("--antialiasing", default="auto",
                    help="RTX anti-aliasing: auto (FXAA on Isaac Lab 2, DLAA on 3) | FXAA | "
                         "DLSS | TAA | DLAA | Off")
    ap.add_argument("--usd", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    return ap


def _policy(spec: str, env):
    if spec in ("expert", "zero"):
        return spec
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit(f"--policy {spec!r}: expected 'expert', 'zero' or module:factory")
    return getattr(importlib.import_module(mod), fn)(env)


def main(argv=None) -> int:
    ap = _args(argv)
    sim_app = None
    pre, _ = ap.parse_known_args(argv)
    if pre.num_envs <= 0:
        ap.error("--num-envs must be positive")
    if pre.num_envs > 1 and pre.plant != "sim":
        ap.error("--num-envs requires --plant sim")
    if pre.episodes <= 0:
        ap.error("--episodes must be positive")
    if pre.max_steps is not None and pre.max_steps <= 0:
        ap.error("--max-steps must be positive")
    out = Path(pre.out)
    if (out / "episodes.jsonl").exists() or (out / "summary.json").exists():
        ap.error("--out already contains results; choose a new directory")
    if pre.plant == "kinematic":
        args = ap.parse_args(argv)
    else:
        from isaaclab.app import AppLauncher

        from isaacswim.launch import prepare_launch
        AppLauncher.add_app_launcher_args(ap)
        ap.set_defaults(device="cuda:0" if pre.num_envs > 1 else "cpu")
        args = ap.parse_args(argv)
        views = [] if args.views == "none" else args.views.split(",")
        args.enable_cameras = bool(views) or args.sonar or args.video
        args.headless = True
        sim_app = AppLauncher(prepare_launch(args, oceansim=args.sonar or args.oceansim)).app

    from isaacswim_bench import KinematicEnv, make_task, rollout, summarize
    from isaacswim_bench.dataset import DemoWriter

    task = make_task(args.task)
    common = dict(control_hz=args.control_hz, action_mode=args.action_mode,
                  sensor_noise=args.sensor_noise, paraphrase=not args.no_paraphrase,
                  current_scale=args.current_scale)
    if args.num_envs > 1:
        from isaacswim_bench.vector_run import run_vector
        run_vector(args, common, _policy)
        sys.stdout.flush()
        os._exit(0)
    if args.plant == "kinematic":
        env = KinematicEnv(task, **common)
    else:
        from isaacswim.assets.bluerov import DEFAULT_USD
        from isaacswim_bench.sim import SimEnv
        views = [] if args.views == "none" else args.views.split(",")
        env = SimEnv(task, usd=args.usd or DEFAULT_USD, device=args.device,
                     views=views, image_size=tuple(args.image_size),
                     water=None if args.water == "none" else args.water,
                     sonar=args.sonar, oceansim=args.oceansim, nav=args.nav,
                     video=args.video, video_height=args.video_height,
                     antialiasing=args.antialiasing,
                     **common)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = _policy(args.policy, env)
    writer = None
    if args.record:
        writer = DemoWriter(args.record, keep_failed=args.keep_failed, env_args={
            "env_name": f"isaacswim_bench/{task.name}", "type": "isaacswim_bench",
            "env_kwargs": {"task": task.name, "plant": args.plant, **common,
                           "views": args.views, "image_size": list(args.image_size),
                           "water": args.water, "sonar": args.sonar,
                           "oceansim": args.oceansim, "nav": args.nav}})

    try:
        reports = []
        with open(out / "episodes.jsonl", "x", encoding="utf-8") as log:
            for k in range(args.episodes):
                seed = args.seed0 + k
                video = None
                on_step = None
                if args.video and args.plant == "sim":
                    from isaacswim.isaaclab import VideoWriter
                    video = VideoWriter(out / f"{task.name}_{seed}.mp4",
                                        fps=int(round(1.0 / env.control_dt)))
                    on_step = lambda e, *_, v=video: v.append(e.video_frame)  # noqa: E731
                t0 = time.perf_counter()
                try:
                    rep = rollout(env, policy, seed=seed, recorder=writer,
                                  max_steps=args.max_steps, on_step=on_step)
                finally:
                    if video is not None:
                        video.close()
                rep["wall_seconds"] = round(time.perf_counter() - t0, 2)
                reports.append(rep)
                log.write(json.dumps(rep) + "\n")
                log.flush()
                print(f"[{task.name} seed {seed}] {'SUCCESS' if rep['success'] else 'fail'}"
                      f" ({rep['failure']})  {rep['sim_seconds']:.1f} s sim, "
                      f"{rep['wall_seconds']:.1f} s wall  -- {rep['instruction']}", flush=True)

        summary = summarize(reports)
        summary.update(policy=args.policy, plant=args.plant, seed0=args.seed0,
                       current_scale=args.current_scale)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: summary[k] for k in ("task", "episodes", "success_rate",
                                                  "success_ci95", "failures")}, indent=2))
        if writer is not None:
            print(f"{writer.num_demos} demos in {args.record}")
    finally:
        if writer is not None:
            writer.close()
    if sim_app is not None:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.exit(code)

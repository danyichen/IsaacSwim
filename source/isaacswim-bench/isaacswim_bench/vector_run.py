from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .rollout import summarize


def _merge_demos(paths, destination):
    import h5py

    if Path(destination).exists():
        raise FileExistsError(f"parallel recording needs a new file: {destination}")
    partial = Path(str(destination) + ".partial")
    owned = False
    try:
        with h5py.File(partial, "x") as target:
            owned = True
            data = target.create_group("data")
            successful, total, count = [], 0, 0
            metadata = None
            for path in paths:
                with h5py.File(path, "r") as source:
                    attrs = dict(source["data"].attrs)
                    if metadata is None:
                        metadata = attrs.get("env_args")
                        data.attrs.update(attrs)
                    elif metadata != attrs.get("env_args"):
                        raise ValueError("recording shards have different environment metadata")
                    for name in sorted(source["data"], key=lambda n: int(n.split("_")[1])):
                        new = f"demo_{count}"
                        source.copy(source["data"][name], data, name=new)
                        group = data[new]
                        total += int(group.attrs["num_samples"])
                        if json.loads(group.attrs["ep_meta"])["report"]["success"]:
                            successful.append(new)
                        count += 1
            data.attrs["total"] = total
            target.create_group("mask").create_dataset(
                "successful", data=np.asarray(successful, dtype="S")
            )
        os.link(partial, destination)
        partial.unlink()
    except BaseException:
        if owned:
            partial.unlink(missing_ok=True)
        raise


def run_vector(args, common, policy_factory):
    from isaacswim.assets.bluerov import DEFAULT_USD

    from .dataset import DemoWriter
    from .sim import VectorSimEnv

    if args.record and Path(args.record).exists():
        raise FileExistsError(f"parallel recording needs a new file: {args.record}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = min(args.num_envs, args.episodes)
    env = VectorSimEnv(
        args.task,
        count,
        device=args.device,
        usd=args.usd or DEFAULT_USD,
        views=() if args.views == "none" else args.views.split(","),
        image_size=tuple(args.image_size),
        water=None if args.water == "none" else args.water,
        sonar=args.sonar,
        oceansim=args.oceansim,
        nav=args.nav,
        video=args.video,
        video_height=args.video_height,
        antialiasing=args.antialiasing,
        **common,
    )
    env.reset(seed=args.seed0)
    policies = [policy_factory(args.policy, child) for child in env.envs]
    recording = Path(args.record) if args.record else None
    if recording and recording.exists():
        raise FileExistsError(f"parallel recording needs a new file: {recording}")
    writers, videos, paths = [], [None] * count, []
    active = np.ones(count, dtype=bool)
    assigned = count
    steps = np.zeros(count, dtype=int)
    started = [time.perf_counter()] * count
    metadata = {
        "env_name": f"isaacswim_bench/{args.task}",
        "type": "isaacswim_bench",
        "env_kwargs": {
            "task": args.task,
            "plant": "sim",
            **common,
            "views": args.views,
            "image_size": list(args.image_size),
            "water": args.water,
            "sonar": args.sonar,
            "oceansim": args.oceansim,
            "nav": args.nav,
        },
    }
    reports = []

    def begin(i):
        child, policy = env.envs[i], policies[i]
        if callable(policy) and hasattr(policy, "reset"):
            policy.reset(child.instruction)
        if writers[i] is not None:
            writers[i].start(child)
        if args.video:
            from isaacswim.isaaclab import VideoWriter

            videos[i] = VideoWriter(
                out / f"{args.task}_{child.seed}.mp4", fps=round(1 / child.control_dt)
            )
        steps[i] = 0
        started[i] = time.perf_counter()

    try:
        for i in range(count):
            writer = None
            if recording:
                path = out / f"recording_slot_{i}.hdf5"
                paths.append(path)
                writer = DemoWriter(path, keep_failed=args.keep_failed, env_args=metadata)
            writers.append(writer)
        for i in range(count):
            begin(i)
        with (out / "episodes.jsonl").open("x", encoding="utf-8") as log:
            while len(reports) < args.episodes:
                observations = env.observations
                actions = []
                for child, obs, policy in zip(env.envs, observations, policies):
                    if policy == "expert":
                        action = child.expert_action()
                    elif policy == "zero":
                        action = child.zero_action()
                    else:
                        action = policy(obs)
                    actions.append(action)
                _, reward, done, truncated, _ = env.step(np.asarray(actions))
                steps += 1
                if int(steps.max()) % 500 == 0:
                    progress = ", ".join(
                        f"slot {i}: seed {child.seed}, {child.world.t:.1f}s"
                        for i, child in enumerate(env.envs)
                        if active[i]
                    )
                    print(
                        f"[{args.task}] {len(reports)}/{args.episodes} episodes; {progress}",
                        flush=True,
                    )
                reset = []
                seeds = []
                for i, child in enumerate(env.envs):
                    if active[i]:
                        if writers[i] is not None:
                            writers[i].add(observations[i], actions[i], reward[i], done[i])
                        if videos[i] is not None:
                            videos[i].append(child.video_frame)
                    limit = args.max_steps is not None and steps[i] >= args.max_steps
                    if not (done[i] or truncated[i] or limit):
                        continue
                    if active[i]:
                        report = child.report()
                        report["truncated"] = not bool(done[i])
                        report["wall_seconds"] = round(time.perf_counter() - started[i], 2)
                        reports.append(report)
                        log.write(json.dumps(report) + "\n")
                        log.flush()
                        print(
                            f"[{args.task} seed {child.seed}] "
                            f"{'SUCCESS' if report['success'] else 'fail'}",
                            flush=True,
                        )
                        if writers[i] is not None:
                            writers[i].end(report)
                        if videos[i] is not None:
                            videos[i].close()
                            videos[i] = None
                        active[i] = assigned < args.episodes
                    reset.append(i)
                    seeds.append(args.seed0 + assigned if active[i] else 2**30 + i)
                    if active[i]:
                        assigned += 1
                if len(reports) == args.episodes:
                    break
                if reset:
                    env.reset(seed=seeds, env_ids=reset)
                    for i in reset:
                        if active[i]:
                            begin(i)
        summary = summarize(reports)
        summary.update(
            policy=args.policy,
            plant="sim",
            seed0=args.seed0,
            current_scale=args.current_scale,
            num_envs=count,
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
        for video in videos:
            if video is not None:
                video.close()
    if recording:
        recording.parent.mkdir(parents=True, exist_ok=True)
        _merge_demos(paths, recording)
        for path in paths:
            path.unlink()
    return summary

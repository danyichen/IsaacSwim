"""Fly the demo reef service course and record the four-view tour video.

Pipeline, valve, coral nursery, instrument reading and dock, starting at the first task.
The prebuilt reef level downloads on first use; --no-reef uses the procedural seabed instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from isaacswim.launch import prepare_launch

ap = argparse.ArgumentParser(description="Render the benchmark course tour.")
ap.add_argument("--out", required=True, help="video output dir, not a synced folder")
ap.add_argument("--reef", default="auto", help="reef level folder; default: download it")
ap.add_argument("--no-reef", action="store_true", help="use the procedural seabed")
ap.add_argument("--from-dock", action="store_true", help="start in the dock, not at the first task")
ap.add_argument("--panel", type=int, default=480, help="pixels per panel side")
ap.add_argument("--layout", choices=("row", "grid"), default="row",
                help="row: four cameras side by side; grid: two by two")
ap.add_argument("--fps", type=int, default=0,
                help="video frame rate; must divide the physics rate; 0: the control rate")
ap.add_argument("--seed", type=int, default=0, help="the episode: its layout and currents")
ap.add_argument("--max-steps", type=int, default=None, help="stop early, for a quick look")
ap.add_argument("--sonar-gain", type=float, default=2.0, help="sonar display gain")
ap.add_argument("--sonar-pitch", type=float, default=20.0, help="sonar tilt below horizontal, deg")
ap.add_argument("--nav", choices=("dvl", "truth"), default="dvl")
ap.add_argument("--antialiasing", default="DLAA",
                choices=("auto", "FXAA", "DLAA", "DLSS", "TAA", "Off"), help="RTX anti-aliasing")
ap.add_argument("--water", default="hazy_ocean",
                help="camera water: hazy_ocean (~30 m visibility), clear_ocean, harbour")
AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(device="cpu")
args = ap.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(prepare_launch(args, oceansim=True)).app

TITLES = {"pipeline_inspect": "pipeline inspection", "valve_turn": "valve manifold",
          "coral_relocate": "coral nursery", "meter_read": "instrument reading",
          "dock": "docking station"}
CAPTIONS = ("Hull camera", "Chase / thruster forces", "Wrist camera", "Imaging sonar")
HULL_EYE = (0.30, 0.0, 0.06)


def main():
    import carb
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from isaacswim_bench.sim import SimEnv
    from isaacswim_bench.sonar_view import render_sonar
    from reef_course import CourseTour

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)

    task = CourseTour()
    task.start_at_first = not args.from_dock
    task.reef = None if args.no_reef else args.reef
    env = SimEnv(task, device=args.device,
                 views=("front", "arm"), image_size=(args.panel, args.panel),
                 video=True, video_close=False, video_height=args.panel,
                 video_specks={"n": 2400, "box": (6.0, 6.0, 2.4), "radius": 0.004},
                 specks_in_views=True, arrows_in_views=True, water=args.water,
                 sonar=True, sonar_pitch=args.sonar_pitch, nav=args.nav, paraphrase=False,
                 antialiasing=args.antialiasing)
    obs, info = env.reset(seed=args.seed)
    print(info["instruction"], flush=True)
    print(f"spawned at {np.round(env.world.pos, 3)} (bay {env.task.bay}), "
          "currents " + ", ".join(f"{k} {np.round(c['mean'][:2], 2)}"
                                  for k, c in info['params']['currents'].items()), flush=True)

    P = args.panel
    s = max(1.0, P / 480)
    head, lab = round(40 * s) // 2 * 2, round(24 * s) // 2 * 2
    cols, rows = (4, 1) if args.layout == "row" else (2, 2)
    size = (cols * P, head + rows * (P + lab))
    sonar_grid = env.sonar_grid
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", round(16 * s))
        label_font = ImageFont.truetype("DejaVuSans.ttf", round(14 * s))
    except OSError:
        title_font = label_font = ImageFont.load_default()

    def board(views, chase, sonar, label, t):
        fan = Image.fromarray(render_sonar(np.squeeze(sonar), sonar_grid,
                                           height=P, gain=args.sonar_gain))
        fan.thumbnail((P, P))
        sonar_rgb = Image.new("RGB", (P, P))
        sonar_rgb.paste(fan, ((P - fan.width) // 2, (P - fan.height) // 2))
        x0 = (chase.shape[1] - chase.shape[0]) // 2
        images = (views["front"], chase[:, x0:x0 + chase.shape[0]], views["arm"],
                  sonar_rgb)
        img = Image.new("RGB", size, (8, 14, 23))
        d = ImageDraw.Draw(img)
        d.text((12, 10), f"IsaacSwim | {TITLES.get(label, label)} | t = {t:5.1f} s",
               font=title_font, fill="white")
        for i, (im, cap) in enumerate(zip(images, CAPTIONS, strict=True)):
            px, py = (i % cols) * P, head + (i // cols) * (P + lab)
            d.text((px + 8, py + 3), cap, font=label_font, fill=(190, 210, 228))
            img.paste(Image.fromarray(np.asarray(im)).resize(
                (P, P), Image.Resampling.BILINEAR), (px, py + lab))
        return img

    fps = args.fps or round(1.0 / env.control_dt)
    every = round(1.0 / (fps * env.physics_dt))
    if abs(every * env.physics_dt * fps - 1.0) > 1e-6:
        raise SystemExit(f"--fps must divide the physics rate, {1.0 / env.physics_dt:.0f} Hz")
    writer = imageio.get_writer(out / "tour.mp4", fps=fps, quality=None, macro_block_size=None,
                                ffmpeg_params=["-crf", "23", "-preset", "slow"])
    env.rig.views["front"].eye = HULL_EYE
    advance = env.showcase.advance
    env.showcase.advance = lambda thrust=None, dt=0.0: advance(thrust, 0.0)
    sonar = env.vehicle.sensors["sonar"]
    t0, n, frames, son_max, last = env.world.t, 0, 0, [], [None]

    def capture():
        nonlocal frames
        advance(env.thrusters.thrust[0], 1.0 / fps)
        env.rig.views["front"].look = (math.cos(env.camera_tilt), 0.0, math.sin(env.camera_tilt))
        views = env.rig.grab(env.sim.render, dt=env.physics_dt)
        chase = env.showcase.grab(env.sim.render, render_dt=env.physics_dt)
        polar = sonar.measure()[0][0].detach().cpu().numpy()
        ex = env.expert
        last[0] = board(views, chase, polar, ex.state if ex else "dock", t0 + n * env.physics_dt)
        if son_max or np.any(polar):
            writer.append_data(np.asarray(last[0]))
            son_max.append(float(np.percentile(polar, 99.8)))
            frames += 1

    def stepped():
        nonlocal n
        n += 1
        if n % every == 0:
            capture()

    env.on_physics_step = stepped
    k = 0
    try:
        capture()
        while True:
            if k % 100 == 0:
                ex = env.expert
                note = (f"task {env.stage + 1}/{len(env.task.sites)}: "
                        f"{ex.detail if ex else 'undock'}  |  "
                        f"nav drift {np.linalg.norm((env.nav.pos - env.world.pos)[:2]):.2f} m")
                print(f"step {k}  t={env.world.t:6.1f}  {ex.state if ex else 'dock':16s} {note}  "
                      f"pos {np.round(env.world.pos, 2)}", flush=True)
                last[0].save(out / f"frame_{k:05d}.jpg", quality=88)
            if args.max_steps is not None and k >= args.max_steps:
                break
            obs, _, terminated, truncated, _ = env.step(env.expert_action())
            k += 1
            if terminated or truncated:
                break
    finally:
        writer.close()
    rep = env.report()
    summary = {**rep, "task": env.task.name, "episodes": 1,
               "success_rate": float(rep["success"]),
               "video_layout": args.layout, "video_size": list(size),
               "video_fps": fps, "video_frames": frames}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({k: rep[k] for k in ("success", "failure", "sim_seconds", "path_m", "nav_drift_m",
                               "sites_done", "done_at", "final_dock_err_m",
                               "expert_timeouts", "roll_max_deg", "pitch_max_deg")},
          flush=True)
    print(f"sonar 99.8th percentile over the video: median {np.median(son_max):.3g}, "
          f"max {np.max(son_max):.3g}", flush=True)
    print(f"wrote {out / 'tour.mp4'} ({k} steps, {frames} frames at {fps} fps)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

"""Render camera and imaging-sonar video from a recorded demonstration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from isaacswim_bench.sonar_view import render_sonar


def render(record, out, *, episode="demo_0", panel=512, gain=2.0):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(record, "r") as file:
        group = file[f"data/{episode}"]
        meta = json.loads(group.attrs["ep_meta"])
        obs = group["obs"]
        keys = ("rgb_front", "rgb_arm", "rgb_down", "sonar")
        for key in keys:
            if key not in obs:
                raise KeyError(f"recording is missing {key}")
        frames = len(group["actions"])
        hz = float(meta["control_hz"])
        if any(len(obs[key]) != frames for key in keys):
            raise ValueError("sensor streams have different frame counts")
        grid = meta.get("sonar_grid")
        if grid is None:
            raise KeyError("recording has no sonar_grid; record it with this version")
        labels = ("Forward camera", "Wrist camera", "Downward inspection camera",
                  "Imaging sonar")
        head, label_h = 48, 28
        board_h = head + 2 * (panel + label_h)
        samples = {int(round(f * (frames - 1))) for f in (0.15, 0.4, 0.6, 0.85)}
        sheet = []
        previous = None
        sonar_changed = 0
        with imageio.get_writer(out, fps=hz, quality=None, ffmpeg_params=["-crf", "23"],
                                macro_block_size=None) as writer:
            for step in range(frames):
                board = Image.new("RGB", (2 * panel, board_h), (8, 14, 23))
                draw = ImageDraw.Draw(board)
                draw.text((12, 8), meta["lang"], fill="white")
                caption = f"Expert | t={step / hz:.2f}s | synchronized sensor observations"
                draw.text((12, 27), caption,
                          fill=(180, 205, 225))
                for i, key in enumerate(keys):
                    x = (i % 2) * panel
                    y = head + (i // 2) * (panel + label_h)
                    draw.text((x + 8, y + 7), labels[i], fill="white")
                    if key == "sonar":
                        raw = obs[key][step]
                        if not np.isfinite(raw).all():
                            raise ValueError(f"nonfinite sonar at frame {step}")
                        if previous is not None and not np.array_equal(previous, raw):
                            sonar_changed += 1
                        previous = raw
                        fan = Image.fromarray(render_sonar(raw, grid, height=panel, gain=gain))
                        fan.thumbnail((panel, panel))
                        image = Image.new("RGB", (panel, panel))
                        image.paste(fan, ((panel - fan.width) // 2, (panel - fan.height) // 2))
                    else:
                        image = Image.fromarray(obs[key][step])
                    board.paste(image.resize((panel, panel), Image.Resampling.BILINEAR),
                                (x, y + label_h))
                writer.append_data(np.asarray(board))
                if step in samples:
                    sheet.append(board.resize((512, board_h // 2)))
        contact = Image.new("RGB", (512 * len(sheet), board_h // 2))
        for i, image in enumerate(sheet):
            contact.paste(image, (512 * i, 0))
        contact.save(out.with_suffix(".png"))
        report = {"frames": frames, "fps": hz, "duration_s": frames / hz,
                  "sonar_shape": list(obs["sonar"].shape[1:]),
                  "sonar_display_gain": gain, "sonar_changed_frames": sonar_changed,
                  "episode": episode, "instruction": meta["lang"]}
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render camera and sonar video from a recording.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--gain", type=float, default=2.0)
    args = parser.parse_args()
    out = Path(args.out)
    render(args.record, out if out.suffix else out.with_suffix(".mp4"), episode=args.episode,
           gain=args.gain)

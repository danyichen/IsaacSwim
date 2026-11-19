"""Display an imaging-sonar image as a fan with range arcs."""

from __future__ import annotations

import math

import numpy as np


def _nice_step(max_range: float) -> float:
    for step in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if max_range / step <= 6:
            return step
    return 100.0


def render_sonar(polar, grid: dict, *, height: int = 320, gain: float = 2.0,
                 gamma: float = 0.6, overlay: bool = True) -> np.ndarray:
    from PIL import Image, ImageDraw

    from isaacswim.oceansim import sonar_fan

    fan = sonar_fan(np.asarray(polar, dtype=np.float32), size=height, **grid).numpy()
    value = np.clip(fan * gain, 0.0, 1.0) ** gamma
    image = Image.fromarray((value * 255).astype(np.uint8)).convert("RGB")
    if not overlay:
        return np.asarray(image)

    n_r, n_a = np.asarray(polar).shape[-2:]
    min_range, range_res = grid["min_range"], grid["range_res"]
    lo, res = grid["min_azimuth"], grid["azimuth_res"]
    max_range = min_range + n_r * range_res
    hi = lo + n_a * res
    w, h = image.size
    px = h / max_range
    cx, cy = w / 2, h
    draw = ImageDraw.Draw(image)
    line = (110, 110, 110)
    text = (190, 190, 190)

    def point(r, a):
        return cx + r * px * math.cos(a), cy - r * px * math.sin(a)

    step = _nice_step(max_range)
    r = step
    while r <= max_range + 1e-6:
        box = (cx - r * px, cy - r * px, cx + r * px, cy + r * px)
        draw.arc(box, start=-math.degrees(hi), end=-math.degrees(lo), fill=line)
        x, y = point(r, hi)
        draw.text((x + 3, y - 12), f"{r:g} m", fill=text)
        r += step
    for a in (lo, hi):
        draw.line([point(min_range, a), point(max_range, a)], fill=line)
    return np.asarray(image)

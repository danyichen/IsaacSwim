from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from isaacswim.isaaclab import _lab

__all__ = ["frame_ready", "grab_frame", "grab_frames", "to_uint8", "VideoWriter"]


def _rgb(camera):
    return _lab.to_torch(camera.data.output["rgb"])


def to_uint8(rgb) -> np.ndarray:
    im = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    if im.dtype != np.uint8:
        im = (np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)
    return im


def frame_ready(camera, *, underwater=None, env: int = 0,
                threshold: float = 8.0) -> bool:
    out = {k: _lab.to_torch(v) for k, v in camera.data.output.items()}
    if float(out["rgb"][env].float().mean()) <= threshold:
        return False
    if underwater is None:
        return True
    rng = out.get(underwater.range_key)
    return rng is not None and float(rng[env].min()) < 100.0


def grab_frame(camera, render: Callable[[], None], *, underwater=None, env: int = 0,
               tries: int = 10, settle: int = 0, dt: float = 0.0,
               threshold: float = 8.0, warn: bool = True) -> np.ndarray:
    ok = False
    for _ in range(max(1, tries)):
        render()
        camera.update(dt, force_recompute=True)
        if frame_ready(camera, underwater=underwater, env=env, threshold=threshold):
            ok = True
            break
    if not ok and warn:
        mean = float(_rgb(camera)[env].float().mean())
        print(f"[IsaacSwim] frame never finished rendering after {tries} passes "
              f"(mean {mean:.1f}/255)", flush=True)
    for _ in range(settle):
        render()
        camera.update(dt, force_recompute=True)
    rgb = underwater.rgb() if underwater is not None else _rgb(camera)
    return to_uint8(rgb[env, ..., :3])


def grab_frames(cameras, render: Callable[[], None], *, underwater=None, env: int = 0,
                tries: int = 12, settle: int = 0, dt: float = 0.0, threshold: float = 8.0,
                warn: bool = True) -> list[np.ndarray]:
    cameras = list(cameras)
    uws = list(underwater) if underwater is not None else [None] * len(cameras)
    ok = False
    for _ in range(max(1, tries)):
        render()
        for cam in cameras:
            cam.update(dt, force_recompute=True)
        if all(frame_ready(c, underwater=u, env=env, threshold=threshold)
               for c, u in zip(cameras, uws, strict=True)):
            ok = True
            break
    if not ok and warn:
        means = [round(float(_rgb(c)[env].float().mean()), 1) for c in cameras]
        print(f"[IsaacSwim] frames never finished rendering after {tries} passes "
              f"(means {means}/255)", flush=True)
    for _ in range(settle):
        render()
        for cam in cameras:
            cam.update(dt, force_recompute=True)
    return [to_uint8((u.rgb() if u is not None else _rgb(c))[env, ..., :3])
            for c, u in zip(cameras, uws, strict=True)]


class VideoWriter:
    def __init__(self, path, *, fps: int = 30, sheet_every: int = 0,
                 sheet_cols: int = 4, crf: int = 23):
        import imageio.v2 as imageio

        self.path = Path(path)
        self._imageio = imageio
        self._writer = imageio.get_writer(self.path, format="FFMPEG", fps=fps, quality=None,
                                          macro_block_size=None,
                                          ffmpeg_params=["-crf", str(int(crf)),
                                                         "-preset", "slow"])
        self.sheet_every = int(sheet_every)
        self.sheet_cols = int(sheet_cols)
        self.sheet: list[np.ndarray] = []
        self.n = 0

    def append(self, frame) -> None:
        im = to_uint8(frame)
        self._writer.append_data(im)
        if self.sheet_every and self.n % self.sheet_every == 0:
            self.sheet.append(im.copy())
        self.n += 1

    def close(self, sheet_path=None) -> Path | None:
        self._writer.close()
        if not (self.sheet and sheet_path):
            return None
        cols = min(self.sheet_cols, len(self.sheet))
        rows = (len(self.sheet) + cols - 1) // cols
        h, w, _ = self.sheet[0].shape
        board = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, im in enumerate(self.sheet):
            r, c = divmod(i, cols)
            board[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
        self._imageio.imwrite(sheet_path, board)
        return Path(sheet_path)

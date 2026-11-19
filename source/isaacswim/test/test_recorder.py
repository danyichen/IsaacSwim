from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.isaaclab.recorder import VideoWriter, frame_ready, grab_frame, to_uint8  # noqa: E402


class LateCamera:
    def __init__(self, ready_after=3, partial=True, range_ready_after=None):
        self.renders = 0
        self.ready_after = ready_after
        self.partial = partial
        self.range_ready_after = ready_after if range_ready_after is None \
            else range_ready_after
        self.updates = 0
        self.data = type("D", (), {"output": {}})()
        self._refresh()

    def render(self):
        self.renders += 1

    def update(self, dt, force_recompute=False):
        assert force_recompute, "a stale buffer is exactly the bug"
        self.updates += 1
        self._refresh()

    def _refresh(self):
        rgb = torch.zeros(1, 8, 8, 3, dtype=torch.uint8)
        if self.renders >= self.ready_after:
            rgb[:] = 60
        elif self.partial and self.renders > 0:
            rgb[0, 0, 0] = 20
        rng = torch.full((1, 8, 8), 1e6 if self.renders < self.range_ready_after
                         else 2.0)
        self.data.output = {"rgb": rgb, "distance_to_camera": rng}


class Underwater:
    range_key = "distance_to_camera"

    def __init__(self, cam):
        self.cam = cam

    def rgb(self):
        return self.cam.data.output["rgb"].float() / 255.0 * 0.5


def test_a_half_written_frame_is_not_ready():
    cam = LateCamera(ready_after=5)
    cam.render()
    cam.update(0.0, force_recompute=True)
    assert cam.data.output["rgb"].max() > 0, "the fake must look partly rendered"
    assert not frame_ready(cam)


def test_grab_waits_for_the_finished_frame():
    cam = LateCamera(ready_after=3)
    im = grab_frame(cam, cam.render)
    assert cam.renders == 3
    assert im.shape == (8, 8, 3) and im.dtype == np.uint8 and im.mean() == 60


def test_the_range_channel_has_to_be_ready_too():
    cam = LateCamera(ready_after=2, range_ready_after=4)
    grab_frame(cam, cam.render, underwater=Underwater(cam))
    assert cam.renders == 4, "accepted colour without range: a flat blue frame"


def test_settle_passes_happen_after_ready():
    cam = LateCamera(ready_after=2)
    grab_frame(cam, cam.render, settle=3)
    assert cam.renders == 5


def test_giving_up_returns_the_last_buffer_and_says_so(capsys):
    cam = LateCamera(ready_after=99)
    im = grab_frame(cam, cam.render, tries=4)
    assert cam.renders == 4 and im.shape == (8, 8, 3)
    assert "never finished" in capsys.readouterr().out


def test_underwater_frames_are_read_through_the_optics():
    cam = LateCamera(ready_after=1)
    im = grab_frame(cam, cam.render, underwater=Underwater(cam))
    assert im.mean() == pytest.approx(int(60 / 255 * 0.5 * 255), abs=1)


def test_to_uint8_accepts_float_and_uint8():
    assert to_uint8(np.full((2, 2, 3), 0.5)).max() == 127
    assert to_uint8(torch.full((2, 2, 3), 200, dtype=torch.uint8)).max() == 200


def test_video_writer_streams_and_writes_a_contact_sheet(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    vw = VideoWriter(tmp_path / "v.mp4", fps=10, sheet_every=2)
    for i in range(5):
        vw.append(np.full((16, 16, 3), i * 40, dtype=np.uint8))
    sheet = vw.close(tmp_path / "sheet.png")
    assert (tmp_path / "v.mp4").stat().st_size > 0
    assert sheet is not None and len(vw.sheet) == 3

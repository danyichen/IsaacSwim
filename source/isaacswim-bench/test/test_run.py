from __future__ import annotations

import subprocess
import sys

import pytest

from isaacswim_bench import KinematicEnv, make_task


def run(*args):
    return subprocess.run([sys.executable, "-m", "isaacswim_bench.run", *map(str, args)],
                          capture_output=True, text=True, timeout=60)


def test_help_is_success():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    assert "--plant" in result.stdout


def test_existing_results_are_not_mixed_or_overwritten(tmp_path):
    log = tmp_path / "episodes.jsonl"
    log.write_text('{"original": true}\n')
    result = run("--task", "dock", "--plant", "kinematic", "--out", tmp_path)
    assert result.returncode == 2
    assert "already contains results" in result.stderr
    assert log.read_text() == '{"original": true}\n'


@pytest.mark.parametrize("kwargs", [
    {"control_hz": 0}, {"control_hz": float("nan")},
    {"sensor_noise": -1}, {"current_scale": float("inf")},
])
def test_invalid_timing_or_noise_is_rejected(kwargs):
    with pytest.raises(ValueError):
        KinematicEnv(make_task("dock"), **kwargs)


def test_demo_metadata_cannot_be_relabelled_on_append(tmp_path):
    pytest.importorskip("h5py")
    from isaacswim_bench.dataset import DemoWriter

    path = tmp_path / "demos.hdf5"
    with DemoWriter(path, env_args={"task": "dock"}):
        pass
    with pytest.raises(ValueError, match="different env_args"):
        DemoWriter(path, env_args={"task": "valve_turn"})
    import h5py
    with h5py.File(path) as f:
        assert '"dock"' in f["data"].attrs["env_args"]


@pytest.mark.parametrize("count", ["0", "-1"])
def test_nonpositive_environment_count_is_rejected(tmp_path, count):
    result = run("--task", "dock", "--num-envs", count, "--out", tmp_path)
    assert result.returncode == 2
    assert "--num-envs must be positive" in result.stderr

from __future__ import annotations

import json

import numpy as np
import pytest

from isaacswim_bench import KinematicEnv, make_task, rollout
from isaacswim_bench.dataset import DemoWriter
from isaacswim_bench.vector_run import _merge_demos

h5py = pytest.importorskip("h5py")


def test_merge_preserves_episodes_observations_actions_and_success_mask(tmp_path):
    paths = []
    metadata = {"task": "dock", "control_hz": 20}
    expected = {}
    for i in range(2):
        path = tmp_path / f"slot{i}.hdf5"
        paths.append(path)
        env = KinematicEnv(make_task("dock"))
        with DemoWriter(path, env_args=metadata, keep_failed=True) as writer:
            rollout(env, "expert" if i == 0 else "zero", seed=i, recorder=writer,
                    max_steps=None if i == 0 else 2)
        with h5py.File(path) as source:
            group = source["data/demo_0"]
            expected[i] = (group["actions"][:], group["obs/proprio"][:])
    destination = tmp_path / "merged.hdf5"
    _merge_demos(paths, destination)
    with h5py.File(destination) as merged:
        assert json.loads(merged["data"].attrs["env_args"]) == metadata
        assert len(merged["data"]) == 2
        assert merged["mask/successful"][:].tolist() == [b"demo_0"]
        total = 0
        for group in merged["data"].values():
            seed = json.loads(group.attrs["ep_meta"])["seed"]
            np.testing.assert_array_equal(group["actions"][:], expected[seed][0])
            np.testing.assert_array_equal(group["obs/proprio"][:], expected[seed][1])
            total += len(group["actions"])
        assert merged["data"].attrs["total"] == total


def test_mismatched_shard_metadata_fails_without_publishing_partial_file(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"slot{i}.hdf5"
        with DemoWriter(path, env_args={"task": str(i)}):
            pass
        paths.append(path)
    destination = tmp_path / "merged.hdf5"
    with pytest.raises(ValueError, match="different environment metadata"):
        _merge_demos(paths, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".hdf5.partial").exists()


def test_existing_partial_recording_is_not_removed(tmp_path):
    destination = tmp_path / "merged.hdf5"
    partial = destination.with_suffix(".hdf5.partial")
    partial.write_bytes(b"another writer owns this")
    with pytest.raises(FileExistsError):
        _merge_demos([], destination)
    assert partial.read_bytes() == b"another writer owns this"
    assert not destination.exists()


def test_existing_destination_is_not_replaced(tmp_path):
    destination = tmp_path / "merged.hdf5"
    destination.write_bytes(b"existing recording")
    with pytest.raises(FileExistsError):
        _merge_demos([], destination)
    assert destination.read_bytes() == b"existing recording"

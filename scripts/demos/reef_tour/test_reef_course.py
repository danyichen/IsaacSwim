from __future__ import annotations

import hashlib
import io
import math
import tarfile

import numpy as np
import pytest

pytest.importorskip("torch")

from isaacswim_bench import KinematicEnv, rollout  # noqa: E402
from isaacswim_bench.dataset import DemoWriter  # noqa: E402
from isaacswim_bench.tasks import make_task  # noqa: E402
from reef_course import CourseTour, fetch_reef, locate_reef  # noqa: E402
from reef_course.course import transit  # noqa: E402
from reef_course.scenery.course import scene_roots  # noqa: E402
from reef_course.service import (  # noqa: E402
    GAUGES,
    PLUG_H,
    PLUG_R,
    STATION_CURRENT,
    STATION_GUST,
    CoralRelocate,
    MeterRead,
)


def test_reports_survive_the_flat_action():
    task = CourseTour()
    reading = {k: 0.5 * (g["lo"] + g["hi"]) for k, g in GAUGES.items()}
    encoded = task.encode_report(reading)
    assert encoded.shape == (len(GAUGES) + 1,)
    assert task.decode_report(encoded) == pytest.approx(reading)
    assert task.decode_report(task.encode_report(None)) is None


def test_course_actions_are_arrays_and_zero_keeps_the_gripper_open():
    env = KinematicEnv(CourseTour())
    env.reset(seed=0)
    assert env.expert_action().shape == (env.action_size,)
    zero = env.zero_action()
    assert zero[env.actions.size - 1] == -1 and zero[env.actions.size] == -1
    assert len(env.action_labels) == env.action_size
    with pytest.raises(ValueError, match="numbers"):
        env.step(np.zeros(env.actions.size))


def test_a_policy_can_finish_the_course_with_the_array_action():
    env = KinematicEnv(CourseTour())
    rep = rollout(env, lambda _obs: env.expert_action(), seed=1)
    assert rep["success"], rep


def test_the_course_records_demonstrations(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "course.hdf5"
    writer = DemoWriter(path)
    env = KinematicEnv(CourseTour())
    assert rollout(env, "expert", seed=0, recorder=writer)["success"]
    writer.close()
    with h5py.File(path) as f:
        actions = f["data/demo_0/actions"][()]
    assert actions.shape[1] == env.action_size
    assert (actions[:, env.actions.size] > 0).any()


def _station():
    task = MeterRead()
    task.seabed_z = -1.5
    params = task.sample(np.random.default_rng(3))
    params.update(panel_xy=[4.6, -1.0], panel_yaw=math.pi)
    return task, params


def _held_world(pos, yaw):
    env = KinematicEnv(make_task("station_keep"))
    env.reset(seed=0)
    world = env.world
    world.pos = np.asarray(pos, dtype=np.float64)
    world.quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return world


def test_station_current_stays_holdable():
    task = MeterRead()
    for seed in range(50):
        params = task.sample(np.random.default_rng(seed))
        assert np.linalg.norm(params["current"]) <= STATION_CURRENT + 1e-9
        assert params["gust"] <= STATION_GUST


def test_every_gauge_has_a_needle_at_its_reading():
    task, params = _station()
    spec = task.scene(params)
    assert {p.name for p in spec.props} == {f"meter_needle_{k}" for k in GAUGES}
    for name, gauge in GAUGES.items():
        center_z = task.seabed_z + 1.0 + gauge["z"]
        fraction = (params[name] - gauge["lo"]) / (gauge["hi"] - gauge["lo"])
        angle = math.radians(225.0 - 270.0 * fraction)
        z = spec.poses[f"meter_needle_{name}"][0][2]
        assert z == pytest.approx(center_z + 0.044 * math.sin(angle))


@pytest.mark.parametrize("name", list(GAUGES))
def test_one_wrong_gauge_fails_the_reading(name):
    task, params = _station()
    target, yaw = task.target(params)
    world = _held_world(target, yaw)
    mem = task.reset(params, world)
    for _ in range(300):
        task.evaluate(world, params, mem, 0.05)
    world.meter_reading = {k: params[k] for k in GAUGES}
    world.meter_reading[name] += 2.0 * GAUGES[name]["tol"]
    assert task.evaluate(world, params, mem, 0.05).failure == "incorrect_meter_reading"


def test_coral_collider_is_the_plug_cylinder():
    task = CoralRelocate()
    task.seabed_z = -1.5
    spec = task.scene(task.sample(np.random.default_rng(0)))
    plugs = [p for p in spec.props if p.kind == "dynamic"]
    assert plugs and all(p.shape == "cylinder" and p.size == (PLUG_R, PLUG_H) for p in plugs)


def test_long_hops_turn_once_and_short_hops_do_not_turn():
    long_hop = transit((0.7, -1.0, -0.55), 0.0, (-0.4, -1.1, -0.72, math.pi))
    assert all(abs(abs(leg.yaw) - math.pi) < 0.2 for leg in long_hop[1:])
    short_hop = transit((0.84, -0.97, -0.5), 0.0, (0.69, -1.0, -0.35, 0.0))
    assert all(leg.yaw == 0.0 for leg in short_hop)


def test_a_stalled_site_fails_instead_of_hanging():
    env = KinematicEnv(CourseTour())
    env.reset(seed=0)
    env._mem["deadline"] = env.world.t - 1.0
    _, _, done, _, info = env.step(env.zero_action())
    assert done and info["failure"].endswith(":timeout")


def test_only_the_shared_scene_has_a_reef_root():
    assert "/World/Reef" in scene_roots("/World", "level")
    assert all("Reef" not in r for r in scene_roots("/World/envs/env_0"))


def _archive(tmp_path, members):
    path = tmp_path / "level.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_reef_download_is_verified_and_cached(tmp_path):
    path, digest = _archive(tmp_path, {"level/reef.usdc": b"usd", "level/sub/a.png": b"png"})
    folder = fetch_reef(url=path.as_uri(), sha256=digest, cache=tmp_path / "cache")
    assert (folder / "sub" / "a.png").read_bytes() == b"png"
    path.unlink()
    assert fetch_reef(url=path.as_uri(), sha256=digest, cache=tmp_path / "cache") == folder
    assert locate_reef(folder) == folder.resolve()


def test_a_bad_reef_download_leaves_nothing_behind(tmp_path):
    path, _ = _archive(tmp_path, {"level/reef.usdc": b"usd"})
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="SHA-256"):
        fetch_reef(url=path.as_uri(), sha256="0" * 64, cache=cache)
    assert list(cache.iterdir()) == []


def test_reef_paths_cannot_escape_the_cache(tmp_path):
    path, digest = _archive(tmp_path, {"level/../../evil": b"x"})
    with pytest.raises(RuntimeError, match="unsafe"):
        fetch_reef(url=path.as_uri(), sha256=digest, cache=tmp_path / "cache")


@pytest.mark.parametrize("seed", range(3))
def test_the_expert_finishes_the_course(seed):
    report = rollout(KinematicEnv(CourseTour()), "expert", seed=seed)
    assert report["success"] and not report["expert_timeouts"], report

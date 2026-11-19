from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from isaacswim.assets import cache_dir  # noqa: E402
from isaacswim_bench import TASKS, KinematicEnv, make_task, rollout, summarize  # noqa: E402
from isaacswim_bench.actions import PilotAction  # noqa: E402
from isaacswim_bench.core import Command  # noqa: E402
from isaacswim_bench.platform import ARM_STOW, TOOL_GRASP, SimpleArm  # noqa: E402
from isaacswim_bench.rollout import wilson  # noqa: E402

ALL = sorted(TASKS)


@pytest.mark.parametrize("name", ALL)
def test_expert_solves_every_task(name):
    env = KinematicEnv(make_task(name))
    for seed in range(3):
        rep = rollout(env, "expert", seed=seed)
        assert rep["success"], (name, seed, rep)
        assert rep["sim_seconds"] < env.task.horizon
        assert not rep["expert_timeouts"], rep["expert_timeouts"]


@pytest.mark.parametrize("name", ALL)
def test_doing_nothing_solves_nothing(name):
    rep = rollout(KinematicEnv(make_task(name)), "zero", seed=1)
    assert not rep["success"]
    assert rep["failure"] == "timeout" and rep["truncated"]


@pytest.mark.parametrize("name", ALL)
def test_the_stage_never_changes_shape(name):
    task = make_task(name)
    first = task.scene(task.sample(np.random.default_rng(0)))
    first.validate()
    for seed in range(1, 25):
        spec = task.scene(task.sample(np.random.default_rng(seed)))
        spec.validate()
        assert first.same_layout(spec), seed


@pytest.mark.parametrize("name", ALL)
def test_every_instruction_formats(name):
    task = make_task(name)
    params = task.sample(np.random.default_rng(4))
    for t in task.templates:
        s = t.format(**params)
        assert "{" not in s and len(s) > 10
    assert task.instruction(params) == task.templates[0].format(**params)


def test_paraphrase_follows_the_seed():
    env = KinematicEnv(make_task("sample_collect"))
    a = [env.reset(seed=s)[1]["instruction"] for s in range(8)]
    b = [env.reset(seed=s)[1]["instruction"] for s in range(8)]
    assert a == b and len(set(a)) > 1
    env = KinematicEnv(make_task("sample_collect"), paraphrase=False)
    assert env.reset(seed=3)[1]["instruction"].startswith("Pick up the")


def _run_with_expert_params(name, seed, edit):
    env = KinematicEnv(make_task(name))
    env.reset(seed=seed)
    fake = edit(dict(env.params))
    env._expert = env.task.expert(fake, env.arm)
    while True:
        _, _, term, trunc, info = env.step(env.expert_action())
        if term or trunc:
            return env.report()


def test_the_wrong_object_fails_target_approach():
    def other(p):
        names = [n for n in p["layout"] if n != p["target"]]
        return {**p, "target": names[0]}
    rep = _run_with_expert_params("target_approach", 2, other)
    assert not rep["success"] and rep["failure"].startswith("wrong_object")


def test_the_wrong_sample_fails_sample_collect():
    def other(p):
        colors = [c for c in p["samples"] if c != p["target"]]
        return {**p, "target": colors[0]}
    rep = _run_with_expert_params("sample_collect", 1, other)
    assert not rep["success"] and rep["failure"].startswith("wrong_sample")


def _follow(env, seed, student, until):
    env.expert_follows = True
    env.reset(seed=seed)
    for step in range(int(env.task.horizon / env.control_dt)):
        label = env.expert_action()
        own = student(env, step) if step < until else None
        _, _, term, trunc, _ = env.step(label if own is None else own)
        if term or trunc:
            break
    return env.report(), env._expert


def test_a_following_expert_still_solves_sample_collect():
    env = KinematicEnv(make_task("sample_collect"))
    for seed in range(3):
        rep, _ = _follow(env, seed, lambda e, k: None, 0)
        assert rep["success"], (seed, rep)


def test_the_following_expert_goes_back_for_a_dropped_sample():
    env = KinematicEnv(make_task("sample_collect"))
    dropped = []

    def butterfingers(e, k):
        a = e._expert and e._expert.state
        if a == "transit" and e._expert.state_t > 1.0 and not dropped:
            dropped.append(k)
        if dropped and k < dropped[0] + 30:
            return np.r_[np.zeros(8), -1.0]
        return None

    rep, expert = _follow(env, 1, butterfingers, 10**9)
    states = [s for _, s in expert.log]
    assert dropped, states
    assert "regrip" in states[states.index("transit"):], states
    assert rep["success"], rep


def test_the_following_expert_lifts_what_the_student_grasped():
    env = KinematicEnv(make_task("sample_collect"))
    env.expert_follows = True
    env.reset(seed=2)
    exp = None
    for _ in range(4000):
        env.step(env.expert_action())
        exp = env._expert
        if exp.state == "close":
            break
    exp.goto("descend", env.world.t)
    exp.start_descent(env.world, "descend")
    for _ in range(40):
        env.step(env.expert_action())
        if exp.state == "lift":
            break
    assert exp.state == "lift", exp.log[-5:]


def test_the_wrong_direction_fails_valve_turn():
    rep = _run_with_expert_params("valve_turn", 0, lambda p: {**p, "sign": -p["sign"]})
    assert not rep["success"] and rep["failure"] == "wrong_direction"


def test_valve_expert_regrasps_when_the_wrist_moves_but_the_handle_slips():
    env = KinematicEnv(make_task("valve_turn"))
    env.reset(seed=0)
    expert, world = env.task.expert(env.params, env.arm), env.world
    expert.arm_cmd = dict(world.arm_q)
    expert.b_goal = world.arm_q["alpha_axis_b"]
    expert._last_valve = world.joints["valve"]
    expert._turned = env.params["sign"] * math.radians(48)
    expert.state, expert.state_t = "turn", 2.0
    expert.act(world)
    assert expert.state == "release"
    assert expert._retry

    expert.latch = world.obj("handle").copy()
    expert.state, expert.state_t = "withdraw", 4.1
    expert.act(world)
    assert expert.state == "deploy"
    assert expert._regrasps == 1
    angle = abs(expert.b_goal - expert.arm_cmd["alpha_axis_b"])
    assert angle == pytest.approx(math.radians(42) + expert.OVERSHOOT)


def test_cutting_the_corner_fails_pipeline_inspect():
    env = KinematicEnv(make_task("pipeline_inspect"))
    env.reset(seed=0)
    end = env.task.vertices(env.params)[-1]
    z = env.task.seabed_z + 1.0
    for _ in range(int(60 / env.control_dt)):
        a = env.actions.encode(Command(pos=np.array([*end, z]), yaw=0.0,
                                       arm=dict(ARM_STOW)))
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    rep = env.report()
    assert not rep["success"] and rep["coverage"] < 0.85


def test_a_recorded_episode_replays():
    env = KinematicEnv(make_task("valve_turn"), sensor_noise=0.0)
    env.reset(seed=5)
    actions = []
    while True:
        a = env.expert_action()
        actions.append(a)
        *_, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    first = env.report()
    env.reset(seed=5)
    for a in actions:
        env.step(a)
    again = env.report()
    assert first["success"] and again["success"]
    assert again["turned_deg"] == pytest.approx(first["turned_deg"])


def _pilot():
    pa = PilotAction(0.05, limits=SimpleArm().limits)
    pa.reset(np.zeros(3), 0.0, ARM_STOW)
    return pa


def test_encode_then_apply_heads_for_the_command():
    pa = _pilot()
    goal = Command(pos=np.array([1.0, -0.5, 0.2]), yaw=0.8, arm=dict(ARM_STOW))
    before = np.linalg.norm(goal.pos - pa.sp.pos)
    sp = pa.apply(pa.encode(goal))
    assert np.linalg.norm(goal.pos - sp.pos) < before
    step = sp.pos / np.linalg.norm(sp.pos)
    assert step @ (goal.pos / np.linalg.norm(goal.pos)) == pytest.approx(1.0, abs=1e-9)


def test_encode_lands_exactly_when_close():
    pa = _pilot()
    goal = Command(pos=np.array([0.004, 0.002, -0.003]), yaw=0.01, arm=dict(ARM_STOW),
                   grip=True)
    sp = pa.apply(pa.encode(goal))
    assert np.allclose(sp.pos, goal.pos) and sp.yaw == pytest.approx(0.01)
    assert sp.grip is True


def test_velocities_are_in_the_heading_frame():
    pa = _pilot()
    pa.sp.yaw = math.pi / 2
    a = np.zeros(9)
    a[0] = 1.0
    sp = pa.apply(a)
    assert sp.pos == pytest.approx([0.0, 0.30 * 0.05, 0.0], abs=1e-12)


def test_the_carrot_is_leashed():
    pa = _pilot()
    a = np.zeros(9)
    a[0] = 1.0
    for _ in range(200):
        sp = pa.apply(a, pos=np.zeros(3), yaw=0.0)
    assert np.linalg.norm(sp.pos) == pytest.approx(pa.leash)


def test_nonfinite_policy_action_cannot_reach_the_plant():
    pa = _pilot()
    before = pa.sp.pos.copy()
    for bad in (float("nan"), float("inf")):
        action = np.zeros(9)
        action[0] = bad
        with pytest.raises(ValueError, match="NaN or infinity"):
            pa.apply(action)
        np.testing.assert_array_equal(pa.sp.pos, before)


def test_arm_setpoints_are_clamped_to_limits():
    pa = _pilot()
    a = np.zeros(9)
    a[4] = 1.0
    for _ in range(2000):
        sp = pa.apply(a)
    assert sp.arm["alpha_axis_e"] == pytest.approx(SimpleArm().limits["alpha_axis_e"][1])


def test_wrench_mode_shape():
    pa = PilotAction(0.05, mode="wrench")
    pa.reset(np.zeros(3), 0.0, ARM_STOW)
    sp = pa.apply(np.r_[np.ones(6), np.zeros(4), 1.0])
    assert sp.wrench.shape == (6,) and sp.wrench[0] > 0 and sp.grip
    with pytest.raises(ValueError):
        pa.apply(np.zeros(9))


def test_simple_arm_round_trips():
    arm = SimpleArm()
    q, info = arm.ik(TOOL_GRASP, roll=1.0, seed=ARM_STOW)
    assert q is not None and info["error"] < 1e-9
    p, R = arm.fk(q)
    assert np.allclose(p, TOOL_GRASP) and np.allclose(R @ R.T, np.eye(3))
    assert R[:, 2] == pytest.approx([0.0, 0.0, -1.0])


def test_observation_has_no_horizontal_position():
    env = KinematicEnv(make_task("dock"))
    obs, _ = env.reset(seed=0)
    assert obs["proprio"].shape == (18,)
    assert set(obs) >= {"depth", "altitude", "imu", "dvl", "arm_q", "gripper",
                        "instruction", "priv_task", "priv_odom"}
    public = [k for k in obs if not k.startswith("priv_") and k != "instruction"]
    env._gen.manual_seed(0)
    base = env.observe()
    env._gen.manual_seed(0)
    env.world.pos[:2] += 5.0
    moved = env.observe()
    for k in public:
        assert np.allclose(base[k], moved[k]), k


def test_wilson_and_summary():
    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.98
    reps = [rollout(KinematicEnv(make_task("station_keep")), "expert", seed=s)
            for s in range(3)]
    s = summarize(reps)
    assert s["episodes"] == 3 and s["success_rate"] == 1.0
    assert "median" in s["hold_err_rms_m"]


def test_demo_writer_round_trip(tmp_path):
    pytest.importorskip("h5py")
    from isaacswim_bench.dataset import DemoWriter, load_demos

    env = KinematicEnv(make_task("target_approach"))
    path = tmp_path / "d.hdf5"
    with DemoWriter(path, env_args={"env_name": "x"}) as w:
        good = rollout(env, "expert", seed=0, recorder=w)
        rollout(env, "zero", seed=1, recorder=w, max_steps=30)
        assert w.num_demos == 1
    demos = list(load_demos(path))
    assert len(demos) == 1
    meta, obs, act = demos[0]
    assert meta["lang"] == good["instruction"] and meta["task"] == "target_approach"
    assert act.shape == (good["steps"], 9) and np.abs(act).max() <= 1.0
    assert obs["proprio"].shape == (good["steps"], 18)

    with DemoWriter(path, keep_failed=True) as w:
        rollout(env, "zero", seed=2, recorder=w, max_steps=10)
        assert w.num_demos == 2
    assert len(list(load_demos(path))) == 1
    assert len(list(load_demos(path, successful_only=False))) == 2


def test_grip_angle_table_is_monotone_and_measured_at_the_ends():
    from isaacswim_bench.platform import GRIP_CLOSED, GRIP_OPEN, grip_angle_for_gap, gripper_open

    assert grip_angle_for_gap(0.087) == pytest.approx(GRIP_OPEN, abs=0.002)
    assert grip_angle_for_gap(0.0) == pytest.approx(GRIP_CLOSED)
    qs = [grip_angle_for_gap(g / 1000) for g in range(0, 88, 5)]
    assert all(a > b for a, b in zip(qs, qs[1:], strict=False))
    assert 0.38 < grip_angle_for_gap(0.045) < 0.46
    assert gripper_open(GRIP_OPEN) and not gripper_open(grip_angle_for_gap(0.045))


URDF = cache_dir() / "bluerov2_heavy_robotiq" / "bluerov2_heavy_robotiq.urdf"


@pytest.mark.skipif(not URDF.is_file(), reason="build the asset: scripts/tools/build_bluerov.py")
@pytest.mark.parametrize("name", ["sample_collect", "valve_turn"])
def test_manipulation_experts_with_the_alpha_kinematics(name):
    from isaacswim.assets.bluerov import AlphaIK
    from isaacswim_bench.platform import TOOL_MOUNT_JOINT, TOOL_OFFSET, TOOL_TIP_LINK

    arm = AlphaIK(URDF, tool_offset=TOOL_OFFSET, tip_link=TOOL_TIP_LINK,
                  mount_joint=TOOL_MOUNT_JOINT)
    env = KinematicEnv(make_task(name), arm=arm)
    for seed in range(3):
        assert rollout(env, "expert", seed=seed)["success"], seed

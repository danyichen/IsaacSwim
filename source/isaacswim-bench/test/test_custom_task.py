from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")

from isaacswim_bench import KinematicEnv, make_task, rollout  # noqa: E402
from isaacswim_bench.core import Task  # noqa: E402


def test_tasks_load_from_a_module_path(tmp_path, monkeypatch):
    (tmp_path / "my_tasks.py").write_text(
        "from isaacswim_bench.core import Task\nclass Mine(Task):\n    name = 'mine'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_tasks", None)
    task = make_task("my_tasks:Mine")
    assert isinstance(task, Task) and task.name == "mine"


def test_the_tutorial_task_is_solved_by_its_expert(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    env = KinematicEnv(make_task("scripts.tutorials.custom_task:BuoyHover"))
    for seed in range(3):
        report = rollout(env, "expert", seed=seed)
        assert report["success"], report

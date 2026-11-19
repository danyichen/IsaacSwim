"""Underwater navigation and manipulation tasks with scripted experts."""

from .core import Command, Prop, SceneSpec, StepResult, Task, World
from .kinematic import KinematicEnv
from .rollout import rollout, summarize
from .tasks import TASKS, make_task

__version__ = "0.1.0"

__all__ = ["__version__", "TASKS", "make_task", "KinematicEnv", "rollout", "summarize", "Task",
           "World", "Command", "SceneSpec", "Prop", "StepResult"]

"""The task definitions and make_task."""

from __future__ import annotations

import importlib

from ..core import Task
from .approach import TargetApproach
from .dock import Dock
from .pipeline import PipelineInspect
from .sample import SampleCollect
from .station_keep import StationKeep
from .valve import ValveTurn

TASKS: dict[str, type[Task]] = {
    t.name: t for t in (StationKeep, PipelineInspect, TargetApproach, Dock,
                        SampleCollect, ValveTurn)
}


def make_task(name: str) -> Task:
    if ":" in name:
        module, _, attribute = name.partition(":")
        return getattr(importlib.import_module(module), attribute)()
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"no task {name!r}; have {sorted(TASKS)} or use module:Class") from None


__all__ = ["TASKS", "make_task", "StationKeep", "PipelineInspect", "TargetApproach",
           "Dock", "SampleCollect", "ValveTurn"]

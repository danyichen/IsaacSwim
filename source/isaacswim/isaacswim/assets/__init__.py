"""Example robot assets, built into ISAACSWIM_CACHE."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

__all__ = ["cache_dir", "isaac_sim_major", "__version__"]


def cache_dir() -> Path:
    root = Path(os.environ.get("ISAACSWIM_CACHE") or Path.home() / ".cache" / "isaacswim")
    return root / "isaacsim6" if isaac_sim_major() >= 6 else root


def isaac_sim_major() -> int:
    try:
        from importlib.metadata import version
        return int(version("isaacsim").split(".")[0])
    except Exception:
        pass
    app = os.environ.get("ISAAC_PATH") or os.environ.get("ISAACSIM_PATH")
    try:
        return int((Path(app) / "VERSION").read_text().split(".")[0])
    except Exception:
        return 5

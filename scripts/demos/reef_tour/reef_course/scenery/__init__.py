"""Procedural seabed, scenery and the prebuilt reef level for the course."""

import os
from pathlib import Path

TEXTURE_DIR = Path(os.environ.get("ISAACSWIM_CACHE", Path.home() / ".cache" / "isaacswim"))
TEXTURE_DIR = TEXTURE_DIR / "textures"

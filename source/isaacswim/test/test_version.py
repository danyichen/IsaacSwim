from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


def _toml_version(path: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    assert m, f"no version in {path}"
    return m.group(1)


def _init_version(path: Path) -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    assert m, f"no __version__ in {path}"
    return m.group(1)


def test_versions_agree():
    root = SOURCE / "isaacswim"
    bench = SOURCE / "isaacswim-bench"
    versions = {
        "pyproject.toml": _toml_version(root / "pyproject.toml"),
        "config/extension.toml": _toml_version(root / "config" / "extension.toml"),
        "__init__.py": _init_version(root / "isaacswim" / "__init__.py"),
        "bench pyproject.toml": _toml_version(bench / "pyproject.toml"),
        "bench __init__.py": _init_version(bench / "isaacswim_bench" / "__init__.py"),
    }
    assert len(set(versions.values())) == 1, versions


from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
PACKAGES = {
    "isaacswim": ROOT / "source" / "isaacswim" / "isaacswim",
    "isaacswim_bench": ROOT / "source" / "isaacswim-bench" / "isaacswim_bench",
}
BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def _blocks():
    if not DOCS.is_dir():
        return []
    return [(path.name, code) for path in sorted(DOCS.glob("*.md"))
            for code in BLOCK.findall(path.read_text(encoding="utf-8"))]


def _source(module: str) -> str:
    top, *rest = module.split(".")
    base = PACKAGES[top].joinpath(*rest)
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if path.name == "__init__.py":
                text += "".join(p.read_text(encoding="utf-8") for p in base.glob("*.py"))
            return text
    raise AssertionError(f"no module {module}")


@pytest.mark.parametrize("page,code", _blocks())
def test_documented_python_parses_and_imports_real_names(page, code):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module.split(".")[0] in PACKAGES:
            text = _source(node.module)
            for alias in node.names:
                assert re.search(rf"\b{alias.name}\b", text), (page, node.module, alias.name)

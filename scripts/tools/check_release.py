from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def markdown_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a\s+(?:name|id)=[\"\']([^\"\']+)', text))
    counts = {}
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if in_fence or not re.match(r"^#{1,6}\s+", line):
            continue
        heading = re.sub(r"^#+\s+|\s+#+\s*$", "", line)
        heading = re.sub(r"<[^>]+>", "", heading)
        slug = re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")
        index = counts.get(slug, 0)
        counts[slug] = index + 1
        anchors.add(f"{slug}-{index}" if index else slug)
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check documentation links and wheel resources.")
    parser.add_argument("--wheels", type=Path)
    args = parser.parse_args()
    errors = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", "build", "dist", "__pycache__"}
               or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts):
            continue
        body = path.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", body)
        targets += re.findall(r'(?:src|srcset|href)="([^\"]+)"', body)
        for target in targets:
            if re.match(r"[a-zA-Z]+:|//", target):
                continue
            relative, _, anchor = unquote(target).partition("#")
            destination = path.parent / relative if relative else path
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
            elif anchor and destination.suffix == ".md":
                if anchor not in markdown_anchors(destination):
                    errors.append(f"{path.relative_to(ROOT)}: missing heading {target}")
    if args.wheels:
        wheels = sorted(args.wheels.glob("*.whl"))
        if len(wheels) != 2:
            errors.append(f"expected two wheels in {args.wheels}, found {len(wheels)}")
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                for required in ("LICENSE", "NOTICE"):
                    if not any(Path(n).name == required for n in names):
                        errors.append(f"{wheel.name}: missing {required}")
                if (wheel.name.startswith("isaacswim-")
                        and "isaacswim/assets/bluerov/robot.xacro" not in names):
                    errors.append(f"{wheel.name}: missing robot.xacro")
    for error in errors:
        print(error)
    if not errors:
        print("Release file checks passed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

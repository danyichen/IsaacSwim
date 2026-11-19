"""Download the prebuilt reef level: python -m reef_course.level."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from isaacswim.assets import cache_dir

REEF_VERSION = "0.1.0"
REEF_NAME = f"isaacswim-reef-{REEF_VERSION}"
REEF_URL = (
    f"https://github.com/danyichen/IsaacSwim/releases/download/v{REEF_VERSION}/{REEF_NAME}.tar.gz"
)
REEF_SHA256 = "dfe972311b9e15ef8150f839da5275dd7b0e38e695d8599b090f3f8516646eb8"


def _download(url: str, path: Path, sha256: str) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as source, path.open("wb") as out:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"{url} does not match its SHA-256")


def _extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if Path(member.name).is_absolute() or ".." in parts:
                raise RuntimeError(f"unsafe path in the reef archive: {member.name}")
            target = dest.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
            else:
                raise RuntimeError(f"unsupported entry in the reef archive: {member.name}")


def fetch_reef(*, url: str = REEF_URL, sha256: str = REEF_SHA256, cache: Path | None = None,
               force: bool = False) -> Path:
    dest = Path(cache or cache_dir()) / REEF_NAME
    if (dest / "reef.usdc").is_file() and not force:
        return dest
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    archive = partial.with_suffix(".tar.gz")
    try:
        print(f"downloading the reef level from {url}", flush=True)
        _download(url, archive, sha256)
        _extract(archive, partial)
        if not (partial / "reef.usdc").is_file():
            raise RuntimeError("the reef archive has no reef.usdc")
        shutil.rmtree(dest, ignore_errors=True)
        partial.rename(dest)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(partial, ignore_errors=True)
    return dest


def locate_reef(value: str | Path) -> Path:
    if str(value) == "auto":
        return fetch_reef()
    folder = Path(value).expanduser().resolve()
    if not (folder / "reef.usdc").is_file():
        raise FileNotFoundError(f"no reef.usdc in {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the prebuilt reef level.")
    parser.add_argument("--force", action="store_true", help="download it again")
    print(fetch_reef(force=parser.parse_args().force))


if __name__ == "__main__":
    main()

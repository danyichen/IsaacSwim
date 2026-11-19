"""Summarise the episodes in an HDF5 demonstration file."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 demonstration files.")
    parser.add_argument("record")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    import h5py

    with h5py.File(args.record, "r") as file:
        data = file["data"]
        names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
        successful = len(file["mask/successful"]) if "mask/successful" in file else 0
        result = {"file": args.record, "episodes": len(names), "successful_episodes": successful,
                  "total_steps": int(data.attrs.get("total", 0)),
                  "env_args": json.loads(data.attrs.get("env_args", "{}")), "examples": []}
        for name in names[:args.limit]:
            episode = data[name]
            meta = json.loads(episode.attrs["ep_meta"])
            result["examples"].append({
                "name": name, "instruction": meta["lang"], "seed": meta["seed"],
                "control_hz": meta["control_hz"], "success": meta["report"]["success"],
                "actions": list(episode["actions"].shape),
                "observations": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                 for key, value in episode["obs"].items()},
            })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

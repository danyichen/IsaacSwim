"""HDF5 demonstrations in robomimic format: writing and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DemoWriter", "load_demos"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


class DemoWriter:
    def __init__(self, path, *, env_args: dict | None = None, keep_failed: bool = False,
                 exclude=(), compress: int = 4):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "a")
        self.data = self.f.require_group("data")
        if env_args is not None:
            metadata = _jsonable(env_args)
            previous = self.data.attrs.get("env_args")
            if previous is not None and json.loads(previous) != metadata:
                self.f.close()
                raise ValueError("cannot append demonstrations with different env_args")
            self.data.attrs["env_args"] = json.dumps(metadata)
        self.keep_failed = keep_failed
        self.exclude = set(exclude)
        self.compress = int(compress)
        self._grp = None
        self._n = 0
        self._next = 1 + max([int(k.split("_")[1]) for k in self.data.keys()
                              if k.startswith("demo_")] or [-1])

    @property
    def num_demos(self) -> int:
        return len([k for k in self.data.keys() if k.startswith("demo_")])

    def start(self, env) -> None:
        self._name = f"demo_{self._next}"
        self._grp = self.data.create_group(self._name)
        self._meta = {"lang": env.instruction, "task": env.task.name, "seed": env.seed,
                      "params": env.params, "action_labels": list(env.action_labels),
                      "control_hz": 1.0 / env.control_dt}
        grid = getattr(env, "sonar_grid", None)
        if grid:
            self._meta["sonar_grid"] = grid
        self._n = 0

    def _append(self, grp, key: str, value) -> None:
        a = np.asarray(value)
        if key not in grp:
            image = a.dtype == np.uint8 and a.ndim >= 2
            kw = {"compression": "gzip", "compression_opts": self.compress} \
                if image and self.compress else {}
            grp.create_dataset(key, shape=(0, *a.shape), maxshape=(None, *a.shape),
                               dtype=a.dtype, chunks=(1 if image else 256, *a.shape), **kw)
        ds = grp[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = a

    def add(self, obs: dict[str, Any], action, reward: float, done: bool) -> None:
        if self._grp is None:
            raise RuntimeError("add() before start()")
        obs_g = self._grp.require_group("obs")
        for k, v in obs.items():
            if k in self.exclude or isinstance(v, str) or v is None:
                continue
            self._append(obs_g, k, v)
        self._append(self._grp, "actions", np.asarray(action, dtype=np.float32))
        self._append(self._grp, "rewards", np.float32(reward))
        self._append(self._grp, "dones", np.uint8(bool(done)))
        self._n += 1

    def end(self, report: dict) -> bool:
        grp, self._grp = self._grp, None
        keep = bool(report.get("success")) or self.keep_failed
        if not keep or self._n == 0:
            del self.data[self._name]
            return False
        grp.attrs["num_samples"] = self._n
        grp.attrs["ep_meta"] = json.dumps(_jsonable({**self._meta, "report": report}))
        self._next += 1
        self.data.attrs["total"] = int(self.data.attrs.get("total", 0)) + self._n
        self._update_mask(report)
        self.f.flush()
        return True

    def _update_mask(self, report: dict) -> None:
        mask = self.f.require_group("mask")
        names = [n.decode() for n in mask["successful"][()]] if "successful" in mask else []
        if report.get("success"):
            names.append(self._name)
        if "successful" in mask:
            del mask["successful"]
        mask.create_dataset("successful", data=np.array(names, dtype="S"))

    def close(self) -> None:
        if self._grp is not None:
            del self.data[self._name]
            self._grp = None
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_demos(path, *, keys=None, successful_only: bool = True):
    import h5py

    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[1]))
        if successful_only and "mask" in f and "successful" in f["mask"]:
            ok = {n.decode() for n in f["mask/successful"][()]}
            names = [n for n in names if n in ok]
        for n in names:
            g = f["data"][n]
            meta = json.loads(g.attrs["ep_meta"])
            obs = {k: g["obs"][k][()] for k in g["obs"].keys()
                   if keys is None or k in keys}
            yield meta, obs, g["actions"][()]

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["rollout", "summarize", "wilson"]


def rollout(env, policy: str | Callable = "expert", *, seed: int | None = None,
            recorder=None, max_steps: int | None = None,
            on_step: Callable | None = None) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    if callable(policy) and hasattr(policy, "reset"):
        policy.reset(info["instruction"])
    if recorder is not None:
        recorder.start(env)
    steps = 0
    while True:
        if policy == "expert":
            action = env.expert_action()
        elif policy == "zero":
            action = env.zero_action()
        else:
            action = np.asarray(policy(obs), dtype=np.float64)
        nxt, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.add(obs, action, reward, terminated)
        if on_step is not None:
            on_step(env, nxt, action, reward, info)
        obs = nxt
        steps += 1
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    report = env.report()
    report["truncated"] = bool(not terminated)
    if recorder is not None:
        recorder.end(report)
    return report


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(reports: list[dict]) -> dict[str, Any]:
    n = len(reports)
    k = sum(bool(r["success"]) for r in reports)
    out: dict[str, Any] = {"task": reports[0]["task"] if reports else None,
                           "episodes": n, "successes": k,
                           "success_rate": k / n if n else float("nan"),
                           "success_ci95": wilson(k, n),
                           "failures": dict(Counter(r["failure"] for r in reports
                                                    if not r["success"]))}
    skip = {"seed", "success", "steps"}
    keys = sorted({key for r in reports for key, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and key not in skip})
    for key in keys:
        vals = np.array([r[key] for r in reports
                         if isinstance(r.get(key), (int, float)) and r[key] is not None],
                        dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            out[key] = {"median": float(med), "iqr": [float(q1), float(q3)]}
    ok = [r["sim_seconds"] for r in reports if r["success"]]
    if ok:
        out["time_to_success_median_s"] = float(np.median(ok))
    return out

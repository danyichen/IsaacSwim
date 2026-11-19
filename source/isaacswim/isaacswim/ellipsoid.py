from __future__ import annotations

import numpy as np

__all__ = ["carlson_rd", "lamb_coefficients", "added_mass", "ellipsoid_volume"]

_C1, _C2, _C3, _C4 = 3.0 / 14.0, 1.0 / 6.0, 9.0 / 22.0, 3.0 / 26.0
_C5, _C6 = 0.25 * _C3, 1.5 * _C4
_ERRTOL = 1e-4
_MAX_ITER = 200


def carlson_rd(x, y, z):
    xt, yt, zt = (np.array(np.broadcast_arrays(x, y, z)[i], dtype=np.float64)
                  for i in range(3))
    if np.any(zt <= 0.0):
        raise ValueError("carlson_rd: z must be strictly positive")
    if np.any(xt < 0.0) or np.any(yt < 0.0):
        raise ValueError("carlson_rd: x and y must be non-negative")

    total = np.zeros_like(xt)
    fac = np.ones_like(xt)
    for _ in range(_MAX_ITER):
        sx, sy, sz = np.sqrt(xt), np.sqrt(yt), np.sqrt(zt)
        lam = sx * (sy + sz) + sy * sz
        total += fac / (sz * (zt + lam))
        fac = 0.25 * fac
        xt = 0.25 * (xt + lam)
        yt = 0.25 * (yt + lam)
        zt = 0.25 * (zt + lam)
        ave = 0.2 * (xt + yt + 3.0 * zt)
        dx, dy, dz = (ave - xt) / ave, (ave - yt) / ave, (ave - zt) / ave
        if np.max(np.abs(np.stack([dx, dy, dz]))) < _ERRTOL:
            break
    else:  # pragma: no cover
        raise RuntimeError("carlson_rd: duplication loop did not converge")

    ea = dx * dy
    eb = dz * dz
    ec = ea - eb
    ed = ea - 6.0 * eb
    ee = ed + ec + ec
    series = (1.0 + ed * (-_C1 + _C5 * ed - _C6 * dz * ee)
              + dz * (_C2 * ee + dz * (-_C3 * ec + dz * _C4 * ea)))
    return 3.0 * total + fac * series / (ave * np.sqrt(ave))


def ellipsoid_volume(semiaxes) -> np.ndarray:
    s = np.asarray(semiaxes, dtype=np.float64)
    return 4.0 / 3.0 * np.pi * s[..., 0] * s[..., 1] * s[..., 2]


def lamb_coefficients(semiaxes) -> np.ndarray:
    s = np.asarray(semiaxes, dtype=np.float64)
    if s.shape[-1] != 3:
        raise ValueError(f"semiaxes must be (..., 3), got {s.shape}")
    if np.any(s <= 0.0):
        raise ValueError("semiaxes must be strictly positive")
    a2, b2, c2 = s[..., 0] ** 2, s[..., 1] ** 2, s[..., 2] ** 2
    abc = s[..., 0] * s[..., 1] * s[..., 2]
    two_thirds_abc = 2.0 / 3.0 * abc
    out = np.stack([
        two_thirds_abc * carlson_rd(b2, c2, a2),
        two_thirds_abc * carlson_rd(c2, a2, b2),
        two_thirds_abc * carlson_rd(a2, b2, c2),
    ], axis=-1)
    err = np.max(np.abs(out.sum(-1) - 2.0))
    if err > 1e-9:
        raise RuntimeError(f"Lamb coefficients do not sum to 2 (off by {err:.2e}); "
                           "the semi-axes are probably degenerate")
    return out


def added_mass(semiaxes, *, mujoco_scaling: bool = True):
    s = np.asarray(semiaxes, dtype=np.float64)
    alpha = lamb_coefficients(s)
    vol = ellipsoid_volume(s)[..., None]

    virtual_mass = alpha / (2.0 - alpha) * vol

    a2, b2, c2 = s[..., 0] ** 2, s[..., 1] ** 2, s[..., 2] ** 2
    al, be, ga = alpha[..., 0], alpha[..., 1], alpha[..., 2]
    pairs = ((b2, c2, be, ga), (c2, a2, ga, al), (a2, b2, al, be))
    kp = []
    for p2, q2, ap, aq in pairs:
        diff = p2 - q2
        num = diff * diff * (aq - ap)
        den = 2.0 * diff + (p2 + q2) * (ap - aq)
        safe = np.where(np.abs(den) > 1e-12 * (p2 + q2), den, 1.0)
        kp.append(np.where(np.abs(den) > 1e-12 * (p2 + q2), num / safe, 0.0))
    kp = np.stack(kp, axis=-1)

    if mujoco_scaling:
        virtual_inertia = kp * vol / 5.0
    else:
        moments = np.stack([b2 + c2, c2 + a2, a2 + b2], axis=-1)
        virtual_inertia = kp * moments * vol / 5.0
    return virtual_mass, virtual_inertia

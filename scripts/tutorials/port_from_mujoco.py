"""Export a MuJoCo model's fluid shapes and check that IsaacSwim reproduces its fluid forces."""

from __future__ import annotations

import sys

import numpy as np
import torch

from isaacswim import HydroWrench
from isaacswim.mjcf import save_json, shapes_from_mjcf, shapes_from_mjmodel

DEMO_XML = """
<mujoco model="demo auv">
  <option density="1025" viscosity="0.00122" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="hull" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="22" diaginertia="0.5 1.2 1.4"/>
      <geom name="hull_drag" type="ellipsoid" size="0.30 0.20 0.15"
            fluidshape="ellipsoid"/>
      <geom name="fin" type="ellipsoid" size="0.12 0.02 0.08" pos="-0.34 0 0.05"
            fluidshape="ellipsoid"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model_from(path_or_none):
    import mujoco

    if path_or_none is None:
        return mujoco.MjModel.from_xml_string(DEMO_XML), "<built-in demo>"
    return mujoco.MjModel.from_xml_path(path_or_none), path_or_none


def extract(path, out):
    shapes, medium = shapes_from_mjcf(path)
    save_json(out, shapes, medium, source=path)
    print(f"{len(shapes)} shapes -> {out}")
    for s in shapes:
        print(f"  {s.body:20s} semiaxes {np.round(s.semiaxes, 4).tolist()}  "
              f"volume {s.volume * 1e3:7.3f} L")


def verify(path=None, n=200, seed=0):
    import mujoco

    model, label = _model_from(path)
    data = mujoco.MjData(model)
    shapes, medium = shapes_from_mjmodel(model)

    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
              for i in range(model.nbody)]
    medium.gravity = (0.0, 0.0, 0.0)
    hydro = HydroWrench(shapes, bodies, medium, dtype=torch.float64)

    body = shapes[0].body
    bid = bodies.index(body)
    if model.body_jntnum[bid] != 1 or model.jnt_type[model.body_jntadr[bid]] != 0:
        print(f"note: body {body!r} has no free joint; the comparison below reads "
              "qfrc_passive for whatever DOFs it does have and may not line up")

    rng = np.random.default_rng(seed)
    worst_f = worst_t = 0.0
    for _ in range(n):
        v = rng.normal(scale=0.5, size=3)
        w = rng.normal(scale=2.0, size=3)
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:3] = v
        data.qvel[3:6] = w
        mujoco.mj_forward(model, data)
        fm, tm = data.qfrc_passive[:3].copy(), data.qfrc_passive[3:6].copy()

        quat = torch.zeros(1, len(bodies), 4, dtype=torch.float64)
        quat[..., 0] = 1.0
        lin = torch.zeros(1, len(bodies), 3, dtype=torch.float64)
        ang = torch.zeros_like(lin)
        lin[0, bid] = torch.tensor(v)
        ang[0, bid] = torch.tensor(w)
        F, T = hydro.compute(quat, lin, ang, pos_w=torch.zeros_like(lin))
        fh, th = F[0, bid].numpy(), T[0, bid].numpy()

        worst_f = max(worst_f, np.linalg.norm(fh - fm) / max(np.linalg.norm(fm), 1e-30))
        worst_t = max(worst_t, np.linalg.norm(th - tm) / max(np.linalg.norm(tm), 1e-30))

    print(f"model: {label}")
    print(f"{len(shapes)} shapes, {n} random states")
    print(f"worst relative error   force {worst_f:.2e}   torque {worst_t:.2e}")
    ok = worst_f < 1e-10 and worst_t < 1e-10
    print("PASS -- the port preserves the physics" if ok else
          "FAIL -- do not train on this until the difference is understood")
    return 0 if ok else 1


def main(argv):
    if not argv:
        return verify()
    if argv[0] == "extract":
        extract(argv[1], argv[2])
        return 0
    if argv[0] == "verify":
        return verify(argv[1] if len(argv) > 1 else None)
    print("usage: port_from_mujoco.py [extract MJCF OUT | verify [JSON]]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

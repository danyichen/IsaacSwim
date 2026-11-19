# Contributing

Issues and pull requests are welcome.

## Setting up

The package installs without Isaac Sim, which is how CI runs:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "source/isaacswim[mujoco,warp,netcdf,cfd]" -e "source/isaacswim-bench[data]" pytest ruff
pytest
ruff check source scripts
```

`isaacswim.isaaclab`, `isaacswim.render`, `isaacswim.oceansim`, `isaacswim.ros2`
and `isaacswim_bench.sim` need Isaac Sim. Test them against doubles where
possible, and say in the pull request what you ran in Isaac Sim.

## Conventions

- Quaternions are `(w, x, y, z)`; units are SI.
- Library modules import Isaac Sim lazily unless they only work inside it.
- Accuracy claims need a test.
- Keep outputs, built assets and downloaded CAD out of the tree; scripts take
  `--out` and assets live in `$ISAACSWIM_CACHE`.
- The version numbers in both packages' `pyproject.toml` and `__init__.py`, and in
  `config/extension.toml`, must agree; `test_version.py` checks this.

## Licensing

Contributions are accepted under Apache-2.0. Add third-party CAD, meshes or
textures only if they are CC0 or your own; fetch anything else at build time.

## Releasing

```bash
python -m pip install build twine
python -m build source/isaacswim --outdir dist
python -m build source/isaacswim-bench --outdir dist
python -m twine check --strict dist/*
python scripts/tools/check_release.py --wheels dist
PYTHONPATH=scripts/demos/reef_tour "${ISAACLAB_PATH}/isaaclab.sh" -p -m isaacswim_bench.run \
    --task reef_course:ReefTour --episodes 12 --views none --out runs/release-demo
```

The last command checks the demo in physics; every episode must succeed. Attach both wheels,
the reef archive and the full-speed video to the GitHub release.

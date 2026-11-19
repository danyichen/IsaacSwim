# Reef tour demo

A complete underwater service mission built on IsaacSwim's public task and scene APIs. It shows
what you can make; it is not part of the installed packages.

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/demos/reef_tour/tour.py --out runs/tour
```

The prebuilt level downloads on first use, is verified and is cached. Use `--reef PATH` for a
local copy or `--no-reef` for a procedural seabed.

`reef_course/course.py` and `service.py` define the mission, and `scenery/` dresses the scene.

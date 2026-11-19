# Cameras and sonar

IsaacSwim describes the water with [OceanSim](https://github.com/umfieldrobotics/OceanSim)'s
parameters, so every camera and sensor on a vehicle sees the same water.

## Water

`OceanSimWater` holds OceanSim's veiling colour, attenuation and backscatter
coefficients and its global scale:

```python
from isaacswim import OceanSimWater

OceanSimWater()
OceanSimWater(veil="seaclear", scale=2.6)
OceanSimWater.jerlov("II")
OceanSimWater.from_uw_param(p)
water.to_uw_param()
```

## Underwater camera

`UnderwaterCamera` applies OceanSim's underwater image model to an Isaac Lab
camera, batched over environments. The camera needs both colour and range:
`data_types=["rgb", "distance_to_image_plane"]`.

```python
from isaacswim.isaaclab import UnderwaterCamera

camera = UnderwaterCamera(scene["camera"], OceanSimWater.jerlov("II"))
rgb = camera.rgb()
```

`isaacswim.oceansim_render.render_frame` renders single frames with OceanSim's
own kernels, caustics included. `isaacswim.render.apply_water_rendering` drives
Isaac Sim's RTX volumetric scattering from the same water. Set it before the
first render.

## Cameras on a vehicle

```python
from isaacswim.isaaclab import CameraRig, ChaseView, LinkView, ShowcaseVideo

wrist = LinkView("wrist_link", eye=(0.1, 0.0, 0.0), look=(1.0, 0.0, 0.0))
rig = CameraRig(robot, {"wrist": wrist}, size=(128, 128), water=water)
video = ShowcaseVideo(robot, close=wrist, water=water, seabed_z=-2.0)
sim.reset()

images = rig.grab(sim.render)
frame = video.grab(sim.render, thrust=thrusters.thrust[0], dt=1 / 20)
```

`LinkView` fixes a camera to a link, `ChaseView` follows the vehicle without
rolling with it, and `CameraRig` renders every view until its frame is finished.
`ShowcaseVideo` adds thrust arrows and particles drifting on the current. Every
render updates every camera, so hiding the arrows for some cameras makes the others
flicker. Either keep them in every view (`arrows_in_views=True`, with the cameras
placed clear of the arrows) or hide them with `video.hide_drawings()` and render
with FXAA.

## OceanSim sensors

`isaacswim.oceansim` mounts OceanSim's sensors on a `Vehicle`, next to
[IsaacSwim's own models](sensors.md). OceanSim's DVL finds bottom lock by casting beams at the
scene, where `DopplerVelocityLog` checks altitude against a flat `seabed_z`. OceanSim's sensors
are not batched: each DVL or barometer adapter reads the first robot of its `Vehicle`, so a fleet
needs one `Vehicle` and one adapter per robot. IsaacSwim's models cover a whole batch in one call,
and `UnderwaterCamera` applies OceanSim's image model to a batched Isaac Lab camera.

```python
import isaacswim.oceansim as osim

sensors = {
    "depth": PressureDepth(),
    "sonar": osim.Sonar("/World/rov/base_link", pitch_deg=30),
    "dvl": osim.DVL("/World/rov/base_link"),
    "baro": osim.Barometer("/World/rov/base_link", medium, as_depth=True),
    "camera": osim.Camera("/World/rov/base_link", OceanSimWater.jerlov("II")),
}
```

Under Isaac Lab:

- Launch with `prepare_launch(args, oceansim=True)` so OceanSim's dependencies load.
- Enable scene queries, `SimulationCfg(enable_scene_query_support=True)`, for the DVL's ray casts.
- Label geometry with `osim.set_sonar_reflectivity(...)`; unlabelled geometry gives no sonar return.
- The sonar measures a polar image, range by bearing. `sonar.fan(image)` converts it
  to the fan an imaging sonar displays, apex at the bottom and starboard on the
  right. `isaacswim_bench.sonar_view.render_sonar` adds gain, range arcs and bearing
  lines for viewing.

[`scripts/demos/bluerov_oceansim.py`](https://github.com/danyichen/IsaacSwim/blob/main/scripts/demos/bluerov_oceansim.py)
flies a vehicle with all four.

# Licensing and third-party material

The project source uses [Apache-2.0](https://github.com/danyichen/IsaacSwim/blob/main/LICENSE). [NOTICE](https://github.com/danyichen/IsaacSwim/blob/main/NOTICE) records
third-party attribution. Installing dependencies or generating a robot model does
not bring those materials under the project license.

| Material | Upstream terms | What this repository includes |
| --- | --- | --- |
| MuJoCo fluid model | Apache-2.0 | A PyTorch reimplementation; attribution in NOTICE |
| Demo reef level (release download) | Apache-2.0, original work; AI-generated limestone and sand albedo maps, other maps and all geometry procedural | Not in this repository or its wheels; fetched by the reef demo |
| OceanSim water constants and image model | BSD-3-Clause | Constants and an implementation; full BSD notice in NOTICE |
| BlueROV2 Heavy description | uvms-simulator declares AGPL-3.0-or-later | A builder that downloads the pinned stock-vehicle files; no mesh files |
| Reach Alpha 5 description and meshes | Reach Robotics' license, plus Reach's written simulation-use authorization (below) | A builder that downloads the pinned files from a source carrying Reach's license; no mesh files |
| Robotiq 2F-85 Isaac Sim asset | Robotiq content BSD-3-Clause; portions from NVIDIA's asset CC BY 4.0 (attribution required, changes stated) | A downloader for the pinned `robotiq/isaacsim_assets` files and conversion code; no USD asset. Generated outputs carry both licenses and `ROBOTIQ_ASSET_SOURCE.txt` |
| T200 performance figures | Published by Blue Robotics | No measurement data; the thruster presets use figures from the published specification and fitted model constants |

The asset builder keeps the downloaded description's license and README beside
its cache and generated vehicle, with an `ASSET_SOURCES.json` record of the source
commit and build modifications. The Robotiq output also records its asset URL and
modifications. These records preserve provenance; they do not grant additional
rights or replace any required corresponding source.

Before sharing generated URDFs, converted meshes, USD files, containers, datasets
or rendered media, verify the rights for every included asset. The fact that CAD
can be downloaded from a public repository does not establish that its uploader
can license it. A rendered video and a redistributable 3D asset can have different
permissions.

Isaac Sim, Isaac Lab and simulator assets are installed separately under their
respective terms.

## Reach Alpha 5

The Alpha 5 description and meshes are downloaded only from
[Robotic-Decision-Making-Lab/alpha](https://github.com/Robotic-Decision-Making-Lab/alpha),
pinned to commit `5e1a6648`, whose
[mesh](https://github.com/Robotic-Decision-Making-Lab/alpha/blob/main/alpha_description/meshes/LICENSE)
and [xacro](https://github.com/Robotic-Decision-Making-Lab/alpha/blob/main/alpha_description/xacro/LICENSE)
license is Reach Robotics'. That license restricts use with other software
without Reach's written authorization, and is separate from the parent
package's MIT declaration. The copy of the Alpha inside uvms-simulator is not
used.

Reach Robotics Pty Ltd has given the IsaacSwim author written authorization to
proceed with simulation use, on these conditions, and may revoke it:

- Download the model only from a source that contains Reach's license file.
  The builder fails if the license is missing or does not match the pinned file.
- No redistribution of the original or converted files. Generated URDFs, OBJ
  meshes, USD files, and containers or datasets that include them must not be
  shared; each user builds them locally.
- Reach's notice and copyright lines stay with the files. Every generated
  vehicle carries `REACH_LICENSE` (verbatim), a copy next to the converted
  meshes, `SOURCE_NOTICES/reach-alpha/` and `REACH_NOTICE.txt`.
- IsaacSwim is not affiliated with, endorsed by or sponsored by Reach Robotics.
- Simulation only: the simulated Alpha 5's performance is not representative of
  the real product's performance.

No manufacturer assets are relicensed by IsaacSwim. For uses beyond these
conditions, contact Reach Robotics.

## Using local checkouts

The BlueROV builder accepts `--upstream-dir /path/to/uvms-simulator` and
`--reach-dir /path/to/alpha` for local checkouts at the pinned commits. Each must
hold the pinned files byte for byte, including their license files; they are
read in place and their notices retained. The Robotiq attachment is a separate
asset download; `--no-robotiq` skips it. See [the build instructions](bluerov.md).
These options do not determine ownership or validate a license grant.

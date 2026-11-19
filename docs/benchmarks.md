# Benchmark methods

How the [home page plots](index.md#benchmarks) were measured, on one Linux machine (RTX 4070,
i7-13700F) with IsaacSwim on Isaac Sim 5.1 / Isaac Lab 2.3, Stonefish 1.5 and HoloOcean 2.3.
IsaacSwim and Stonefish fly the same bare BlueROV2 Heavy. HoloOcean flies its built-in BlueROV2.

Each timed job ran alone after a discarded warm-up, with engines interleaved. Physics results are
the median of three runs up to 64 vehicles and one run above.

| Plot | Workload | Score |
|---|---|---|
| **Physics** | 120 Hz. IsaacSwim and Stonefish hold station with IMU, DVL, pressure and odometry. HoloOcean runs constant thrust with one dynamics sensor. | Step time, without controller time. IsaacSwim shows the faster of CPU and GPU at each fleet size. |
| **Startup** | Each engine's native harness, with warm caches. | Time to the first step. Harness timers start at different points. |
| **Cameras** | One RGB camera per vehicle. | Highest real-time factor at which nearly all frames arrive. Hollow markers delivered too few frames. Crosses crashed or ran out of GPU resources. |
| **Accuracy** | Open-loop replay of recorded commands from [Marinarium](https://arxiv.org/abs/2602.23053) BlueROV2 Heavy runs, starting from each valid state in the last fifth of a run. | Error in position, attitude and velocity two seconds ahead. |

Nothing is fitted to the tank data, and no engine models the tether or battery sag.

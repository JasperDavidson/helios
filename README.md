Helios
A C++ task-graph scheduler for real-time LiDAR processing in autonomous racing.

Problem
Autonomous race cars must process millions of LiDAR points per frame with a sub-50ms latency budget. Manually pipelining the required stages (filtering, ground-plane removal, clustering) across a CPU and GPU is brittle, hard to debug, and difficult to optimize.

Solution
Helios streamlines this by representing the entire perception pipeline as a dependency graph. We define the LiDAR processing stages as tasks, and the runtime automatically orchestrates their parallel execution, managing all data movement and synchronization between the CPU and GPU.

Key Features
Declarative Pipeline Definition: Define the multi-stage LiDAR pipeline once and let the runtime handle the low-level execution details.

Full Kernel Control: Write your own highly-optimized CUDA/Metal kernels for tasks like point cloud filtering, and let Helios handle the plumbing.

Backend-Agnostic: Develop the pipeline on a Metal-based laptop and deploy on a CUDA-based in-car computer with zero changes to the graph logic.

Status
Currently building the core runtime and Metal executor. The immediate goal is to execute a simple, two-stage LiDAR filtering pipeline on a public dataset (e.g., KITTI) to validate the core architecture.

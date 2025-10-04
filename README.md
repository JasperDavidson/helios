(WIP)

A description of the explanation behind each system component and architecture decision is available in docs/

Helios is a C++ framework I'm building to simplify high-performance programming on systems with both CPUs and GPUs. It also serves as a performance simulator for exploring future hardware-software co-designs.

Problem:
Modern computer hardware is incredibly powerful, offering a diverse team of specialized processors on a single chip. However, writing software that efficiently utilizes this hardware is painfully complex. Developers are forced to manually manage low-level details like data transfers, synchronization, and vendor-specific APIs (like CUDA or Metal), which is slow, error-prone, and not portable. This creates a massive gap between the hardware's potential performance and what most applications can actually achieve.

Helios bridges this gap by providing two key components:
- A Simple Task-Graph API: A developer can describe their complex, parallel workload as a simple graph of dependencies, focusing on what they want to compute.

- An Intelligent Runtime: The Helios runtime takes this graph and orchestrates its execution across all available hardware. It automates the difficult parts—like data management and synchronization—and uses a cost model to make smart, dynamic decisions about where to run each task to maximize performance.

Key Features:
- Backend-Agnostic Design: The scheduler is completely decoupled from the GPU programming model via an abstract interface. The initial implementation uses Apple's Metal for the GPU backend, but it's designed to easily support CUDA or SYCL in the future.

- Dynamic Placement: The long-term goal is for the scheduler to use its cost models to intelligently decide where to run tasks at runtime, adapting to system load and data locality to squeeze out maximum performance.

Current Status & End Goal:\
The core framework is currently under development.

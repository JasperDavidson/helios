Interface should be **agnostic of GPU backend**
Separate executors can be defined for each backend that all implement this interface
### Key needs
- Executing kernels agnostic of GPU backend
	- Some sort of "execute kernel" function that takes in an identifier (enum? string?) and communicates with GPU backend to load that kernel
- Sending data back and forth between CPU and GPU
	- Allocating buffers on the GPU will be important as well for this

(Insert GPU language here)Interface objects will hold onto whatever information is needed for interaction with the GPUs
- E.g. MetalExecutor has `MTLDevice` and `MTLCommandQueue` as member variables

Will want an `execute_kernel` function
- Takes in the name of a kernel to execute and communicates through the GPU backend to find the kernel and proceed from there

### Interface Details
#### `GPUBufferHandle`
- Struct to contain a reference to GPU buffers
	- Likely just contains an integer ID (?)
- In each backend these handles will map to backend specific buffer objects (e.g. `MetalBuffer`s)
#### `GPUError`
- Enum that contains various results of operations to indicate success/failure
- E.g. `GPUSuccess, GPUError, GPUBufferMemoryExceeded, GPUBufferNotFound, etc.`
#### Methods
- `allocate_buffer(buffer_size) -> GPUBufferHandle`
- `deallocate_buffer(GPUBufferHandle)`
	- Deallocates buffers on the GPU when they're usage is complete to prevent memory leaks
	- Handled by helios as it detects buffers that no longer have buffer depedencies
- `copy_to_device(cpu_address*, gpu_buffer_handle, data_size)`
- `copy_from_device(cpu_address*, gpu_buffer_handle, data_size)`
	- Maybe make the cpu_address a `vector/span` type so it has a guaranteed size?
- `execute_kernel(kernel_name, (list of GPUBufferHandle), grid_size)`
	- Search the backend specific hashmap for a kernel with the given name
	- If found, check if the kernel has been updated and use it if not
		- If the kernel has not been updated, or a cache miss occurs, create the kernel for the specific backend, store it, and use it
		- The kernel function should be found in the generated `.lib` file for each backend
- `synchronize()`
	- Prevent the GPU from executing any new tasks until all tasks are executed

#### Design Notes
- Helios should only ever interact with the `IGpuExecutor` interface
	- CUDA, Metal, etc. backends remain hidden
- Helios should eventually dynamically handle `copy` tasks in both directions (e.g. if a CPU task A stores some data on RAM and a GPU task C requires that data, automatically create a copy task B that transfers that data in the graph)
	- To do this the dynamic Helios will also need a parameter noting that the memory required is already on the GPU and provide buffers or provide an address to CPU memory to operate on
	- For now the user just will just manually add copy requests to the graph

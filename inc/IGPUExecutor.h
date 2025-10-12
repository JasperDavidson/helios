#ifndef IGPU_H
#define IGPU_H

#include <span>

// TODO: Add more options as interface is built out
enum class GPUState { GPUSuccess, GhostBuffer, GPUFailure };

class GPUBufferHandle {
public:
  int ID;

  GPUBufferHandle(int ID) : ID(ID) {};
};

class IGPUExecutor {
public:
  // Allocating/freeing buffer memory on the GPU for kernel tasks
  GPUBufferHandle virtual allocate_buffer(std::uint32_t buffer_size) = 0;
  GPUState virtual deallocate_buffer(const GPUBufferHandle &buffer_handle) = 0;

  // Sending memory between devices for task completion/after task completion
  GPUState virtual copy_to_device(std::span<const std::byte> data_mem,
                                  const GPUBufferHandle &buffer_handle,
                                  std::uint32_t data_size) = 0;
  GPUState virtual copy_from_device(std::span<std::byte> data_mem,
                                    const GPUBufferHandle &buffer_handle,
                                    std::uint32_t data_size) = 0;

  // Maybe a string isn't the best hash?
  GPUState virtual execute_kernel(
      const std::string &kernel_name,
      const std::vector<GPUBufferHandle> &buffer_handles,
      const std::vector<int> &kernel_size) = 0;

  // Prevents more GPU tasks from being added until all current ones are
  // complete
  GPUState virtual synchronize() = 0;

  virtual ~IGPUExecutor() = default;
};

#endif

#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

#include <any>
#include <functional>
#include <span>
#include <unordered_map>

// MemoryHint - How the data stored in the buffer will be treated throughout the lifetime of a task on a CPU/GPU level
//  - Enables optimizations with private memory on the GPU, guarantee that only it has access to it
enum class MemoryHint { DeviceLocal, HostVisible };

class GPUBufferHandle {
  public:
    int id;
    MemoryHint mem_hint;

    GPUBufferHandle() = default;
    GPUBufferHandle(int ID, MemoryHint mem_hint) : id(ID), mem_hint(mem_hint) {};
    bool operator==(const GPUBufferHandle &other) const { return this->id == other.id; }
};

// GPUBufferHandle objects have effective hashes already since they store a unique ID
namespace std {
template <> struct std::hash<GPUBufferHandle> {
    std::size_t operator()(const GPUBufferHandle &buffer_handle) const noexcept {
        return std::hash<int>{}(buffer_handle.id);
    }
};
} // namespace std

// Templated DataHandle allows for associating types needed for std::any casts
template <typename T> class DataHandle {
  public:
    int id;
    DataHandle(int id) : id(id) {};
};

enum class DataUsage { ReadWrite, ReadOnly };

struct DataEntry {
    std::any data;
    size_t byte_size;

    // GPU Memory access hint
    MemoryHint mem_hint;

    // Parameter describing how data will be accessed (default is ReadWrite for safety)
    DataUsage data_usage;

    // Generic getters for accessing data memory
    std::function<std::span<const std::byte>()> const_data_accessor;
    std::function<std::span<std::byte>()> raw_data_accessor;
};

// DataManager object allows for caching of DataHandles to their actual objects
class DataManager {
  public:
    template <typename T> T get_data(DataHandle<T> data_handle) const {
        return std::any_cast<T>(data_map.at(data_handle.ID).data);
    }
    template <typename T>
    DataHandle<T> create_data_handle(T &&data, const DataUsage &data_usage = DataUsage::ReadWrite,
                                     const MemoryHint &mem_hint = MemoryHint::DeviceLocal);

    // Constructor for kernel output data of variable size
    // Will either use the max input size or requires user to set up buffer counting
    template <typename T>
    DataHandle<T> create_date_handle(const DataUsage &data_usage = DataUsage::ReadWrite,
                                     const MemoryHint &mem_hint = MemoryHint::HostVisible, size_t byte_size = 0);

    void store_data(int data_id, std::any new_data) { data_map.at(data_id).data = new_data; };

    std::span<const std::byte> get_span(int ID) const { return data_map.at(ID).const_data_accessor(); };
    std::span<std::byte> get_span_mut(int ID);
    int get_data_length(int ID) const { return data_map.at(ID).byte_size; };
    const MemoryHint &get_mem_hint(int ID) const { return data_map.at(ID).mem_hint; };
    const DataUsage &get_buffer_usage(int ID) const { return data_map.at(ID).data_usage; };
    const std::vector<DataEntry> &get_device_local_tasks() const { return device_local_tasks_; };

  private:
    std::unordered_map<int, DataEntry> data_map;
    std::vector<DataEntry> device_local_tasks_;
    size_t id_counter = 0;
};

#endif

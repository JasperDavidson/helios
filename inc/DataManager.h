#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

#include "IGPUExecutor.h"
#include <any>
#include <unordered_map>

// Templated DataHandle allows for associating types needed for std::any casts
template <typename T> class DataHandle {
  public:
    int ID;
    DataHandle(int ID) : ID(ID) {};
};

struct DataEntry {
    std::any data;
    size_t byte_size;

    // GPU Memory access hint
    MemoryHint mem_hint;

    // Parameter for maintaining dynamic GPU output if user wishes
    bool buffer_count = false;

    // Parameter for how data will be accessed
    BufferUsage buffer_usage;

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
    DataHandle<T> create_data_handle(T &&data, const BufferUsage &buffer_usage = BufferUsage::ReadWrite,
                                     const MemoryHint &mem_hint = MemoryHint::DeviceLocal);
    template <typename T>
    DataHandle<T> create_date_handle(const BufferUsage &buffer_usage = BufferUsage::ReadWrite,
                                     const MemoryHint &mem_hint = MemoryHint::HostVisible, size_t byte_size = 0,
                                     bool buffer_count = false);

    void store_data(int data_id, std::any new_data) { data_map.at(data_id).data = new_data; };

    std::span<const std::byte> get_span(int ID) const { return data_map.at(ID).const_data_accessor(); };
    std::span<std::byte> get_span_mut(int ID);
    int get_data_length(int ID) const { return data_map.at(ID).byte_size; };
    bool get_buffer_count_request(int ID) const { return data_map.at(ID).buffer_count; };
    MemoryHint get_mem_hint(int ID) const { return data_map.at(ID).mem_hint; };
    BufferUsage get_buffer_usage(int ID) const { return data_map.at(ID).buffer_usage; };

  private:
    std::unordered_map<int, DataEntry> data_map;
    size_t id_counter = 0;
};

#endif

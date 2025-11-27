#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

#include "TypeTraits.h"
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <unordered_map>

// MemoryHint - How the data stored in the buffer will be treated throughout the lifetime of a task on a CPU/GPU level
//  - Enables optimizations with private memory on the GPU, guarantee that only it has access to it
enum class MemoryHint { DeviceLocal, Unified, HostVisible };

class GPUBufferHandle {
  public:
    size_t id;
    size_t size;
    MemoryHint mem_hint;
    size_t mem_offset;

    GPUBufferHandle() = default;
    GPUBufferHandle(int id, MemoryHint mem_hint, size_t mem_offset, size_t size)
        : id(id), mem_hint(mem_hint), mem_offset(mem_offset), size(size) {};
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
};

enum class DataUsage { ReadWrite, ReadOnly };

struct DataEntry {
    std::any data;
    bool alias = false;

    // size of the data_entry as a whole
    size_t byte_size;
    // size of the type (e.g. int if std::vector<int>)
    size_t type_size;

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
    size_t get_type_size(int data_id) { return data_map.at(data_id).type_size; }

    template <typename T> T &get_data(DataHandle<T> data_handle) {
        auto &any_data = data_map.at(data_handle.id).data;

        if (any_data.type() == typeid(std::shared_ptr<T>)) {
            auto &any_ptr = std::any_cast<std::shared_ptr<T> &>(any_data);
            return *any_ptr;
        } else if (any_data.type() == typeid(std::reference_wrapper<T>)) {
            return std::any_cast<std::reference_wrapper<T>>(any_data).get();
        }

        throw std::runtime_error("Trying to match data handle ID to mismatched type!");
    }

    template <typename T>
    DataHandle<T> create_data_handle(T data, const DataUsage &data_usage = DataUsage::ReadWrite,
                                     const MemoryHint &mem_hint = MemoryHint::HostVisible) {
        DataEntry entry;
        DataHandle<T> data_handle;

        // Move the data onto the heap immediately via shared pointer
        // This verifies the address of the data never changes even if entry is copied/moved
        auto data_ptr = std::make_shared<T>(std::move(data));
        entry.data = data_ptr;
        T *raw_ptr = data_ptr.get();

        entry.mem_hint = mem_hint;
        entry.data_usage = data_usage;

        // If true then T is a container type, else its size can be found through sizeof() at compile time
        if constexpr (ContiguousContainer<T>) {
            size_t byte_size = raw_ptr->size() * sizeof(typename T::value_type);
            entry.byte_size = byte_size;
            entry.type_size = sizeof(typename T::value_type);

            entry.const_data_accessor = [raw_ptr, byte_size]() {
                return std::span<const std::byte>(reinterpret_cast<const std::byte *>(raw_ptr->data()), byte_size);
            };

            if (data_usage == DataUsage::ReadWrite) {
                entry.raw_data_accessor = [raw_ptr, byte_size]() {
                    return std::span<std::byte>(reinterpret_cast<std::byte *>(raw_ptr->data()), byte_size);
                };
            }
        } else {
            size_t byte_size = sizeof(T);
            entry.byte_size = byte_size;
            entry.type_size = sizeof(T);

            entry.const_data_accessor = [raw_ptr, byte_size]() {
                return std::span<const std::byte>(reinterpret_cast<const std::byte *>(raw_ptr), byte_size);
            };

            if (data_usage == DataUsage::ReadWrite) {
                entry.raw_data_accessor = [raw_ptr, byte_size]() {
                    return std::span<std::byte>(reinterpret_cast<std::byte *>(raw_ptr), byte_size);
                };
            }
        }

        data_handle.id = id_counter++;
        data_map[data_handle.id] = entry;

        if (mem_hint == MemoryHint::DeviceLocal) {
            device_local_tasks_.push_back(entry);
        }

        return data_handle;
    }

    template <typename T>
    DataHandle<T> create_ref_handle(T *data, const DataUsage &data_usage = DataUsage::ReadWrite,
                                    const MemoryHint &mem_hint = MemoryHint::HostVisible) {
        DataEntry entry;
        DataHandle<T> data_handle;

        entry.alias = true;
        entry.data = std::make_any<std::reference_wrapper<T>>(std::ref(*data));

        entry.mem_hint = mem_hint;
        entry.data_usage = data_usage;

        // If true then T is a container type, else it's size can be found through sizeof() at compile time
        if constexpr (ContiguousContainer<T>) {
            size_t byte_size = data->size() * sizeof(typename T::value_type);
            entry.byte_size = byte_size;
            entry.type_size = sizeof(typename T::value_type);

            entry.const_data_accessor = [data, byte_size]() {
                return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data->data()), byte_size);
            };

            if (data_usage == DataUsage::ReadWrite) {
                entry.raw_data_accessor = [data, byte_size]() {
                    return std::span<std::byte>(reinterpret_cast<std::byte *>(data->data()), byte_size);
                };
            }
        } else {
            size_t byte_size = sizeof(T);
            entry.byte_size = byte_size;
            entry.type_size = sizeof(T);

            entry.const_data_accessor = [data, byte_size]() {
                return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data), byte_size);
            };

            if (data_usage == DataUsage::ReadWrite) {
                entry.raw_data_accessor = [data, byte_size]() {
                    return std::span<std::byte>(reinterpret_cast<std::byte *>(data), byte_size);
                };
            }
        }

        data_handle.id = id_counter++;
        data_map[data_handle.id] = entry;

        if (mem_hint == MemoryHint::DeviceLocal) {
            device_local_tasks_.push_back(entry);
        }

        return data_handle;
    }

    // NOTE: Effectively acts as a "placeholder" handle. When the data is passed back from the GPU, create a new "real"
    // data handle as above, and replace this one's position in the map
    // - Maintains a clear structure since the user always interaces with DataHandle objects
    template <typename T>
    DataHandle<T> create_variable_kernel_handle(const DataUsage &buffer_usage, const MemoryHint &mem_hint,
                                                size_t byte_size) {
        DataEntry entry;
        DataHandle<T> data_handle;

        entry.mem_hint = mem_hint;
        entry.data_usage = buffer_usage;
        entry.byte_size = byte_size;

        data_handle.id = id_counter++;

        if (mem_hint == MemoryHint::DeviceLocal) {
            device_local_tasks_.push_back(entry);
        }

        data_map[data_handle.id] = entry;

        return data_handle;
    }

    void store_data(int data_id, std::span<std::byte> new_bytes) {
        auto dest_span = get_span_mut(data_id);
        std::copy(new_bytes.begin(), new_bytes.end(), dest_span.begin());
    };

    template <typename T> void store_data(int data_id, const T &new_data) {
        auto raw_data = get_span_mut(data_id);
        T *dest_object = reinterpret_cast<T *>(raw_data.data());
        *dest_object = new_data;
    }

    template <typename T> void store_data(int data_id, T &&new_data) {
        using U = std::decay_t<T>;
        auto raw_data = get_span_mut(data_id);
        U *dest_object = reinterpret_cast<U *>(raw_data.data());
        *dest_object = std::forward<T>(new_data);
    };

    std::span<const std::byte> get_span(int data_id) const { return data_map.at(data_id).const_data_accessor(); };
    std::span<std::byte> get_span_mut(int data_id);
    int get_data_length(int data_id) const { return data_map.at(data_id).byte_size; };
    const MemoryHint &get_mem_hint(int data_id) const { return data_map.at(data_id).mem_hint; };
    const DataUsage &get_data_usage(int data_id) const { return data_map.at(data_id).data_usage; };
    const std::vector<DataEntry> &get_device_local_tasks() const { return device_local_tasks_; };

  private:
    std::unordered_map<int, DataEntry> data_map;
    std::vector<DataEntry> device_local_tasks_;
    size_t id_counter = 0;
};

#endif

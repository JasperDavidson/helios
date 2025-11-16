#include "DataManager.h"
#include "TypeTraits.h"
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>

std::span<std::byte> DataManager::get_span_mut(int data_id) {
    if (data_map.at(data_id).data_usage != DataUsage::ReadWrite) {
        throw std::runtime_error("Attempted to fetch mutable span into read-only data");
    }

    return data_map.at(data_id).raw_data_accessor();
};

void DataManager::store_data(int data_id, std::any new_data) {
    auto raw_data = get_span_mut(data_id);
    memcpy(raw_data.data(), &new_data, raw_data.size());
};

template <typename T>
DataHandle<T> DataManager::create_data_handle(T data, const DataUsage &data_usage, const MemoryHint &mem_hint) {
    DataEntry entry;
    DataHandle<T> data_handle;

    auto data_ptr = std::make_unique<T>(std::move(data));

    entry.data = data_ptr;
    entry.mem_hint = mem_hint;
    entry.data_usage = data_usage;

    // If true then T is a container type, else it's size can be found through sizeof() at compile time
    if constexpr (isContiguousContainer<T>::value) {
        size_t byte_size = data_ptr->size() * sizeof(typename T::value_type);
        entry.byte_size = byte_size;
        entry.const_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data_ptr->data()), byte_size);
        };

        if (data_usage == DataUsage::ReadWrite) {
            entry.raw_data_accessor = [data_ptr, byte_size]() {
                return std::span<std::byte>(reinterpret_cast<std::byte *>(data_ptr->data()), byte_size);
            };
        }
    } else {
        size_t byte_size = sizeof(T);
        entry.byte_size = byte_size;
        entry.const_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data_ptr), byte_size);
        };

        if (data_usage == DataUsage::ReadWrite) {
            entry.raw_data_accessor = [data_ptr, byte_size]() {
                return std::span<std::byte>(reinterpret_cast<std::byte *>(*data_ptr, byte_size));
            };
        }
    }

    data_handle.ID = id_counter++;
    data_map[data_handle.ID] = entry;

    if (mem_hint == MemoryHint::DeviceLocal) {
        device_local_tasks_.push_back(entry);
    }

    return data_handle;
}

template <typename T>
DataHandle<T> DataManager::create_ref_handle(T *data, const DataUsage &data_usage, const MemoryHint &mem_hint) {
    DataEntry entry;
    DataHandle<T> data_handle;

    auto data_ptr = data;
    entry.alias = true;

    entry.data = data_ptr;
    entry.mem_hint = mem_hint;
    entry.data_usage = data_usage;

    // If true then T is a container type, else it's size can be found through sizeof() at compile time
    if constexpr (isContiguousContainer<T>::value) {
        size_t byte_size = data_ptr->size() * sizeof(typename T::value_type);
        entry.byte_size = byte_size;
        entry.const_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data_ptr->data()), byte_size);
        };

        if (data_usage == DataUsage::ReadWrite) {
            entry.raw_data_accessor = [data_ptr, byte_size]() {
                return std::span<std::byte>(reinterpret_cast<std::byte *>(data_ptr->data()), byte_size);
            };
        }
    } else {
        size_t byte_size = sizeof(T);
        entry.byte_size = byte_size;
        entry.const_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data_ptr), byte_size);
        };

        if (data_usage == DataUsage::ReadWrite) {
            entry.raw_data_accessor = [data_ptr, byte_size]() {
                return std::span<std::byte>(reinterpret_cast<std::byte *>(data_ptr, byte_size));
            };
        }
    }

    data_handle.ID = id_counter++;
    data_map[data_handle.ID] = entry;

    if (mem_hint == MemoryHint::DeviceLocal) {
        device_local_tasks_.push_back(entry);
    }

    return data_handle;
}

// NOTE: Effectively acts as a "placeholder" handle. When the data is passed back from the GPU, create a new "real" data
// handle as above, and replace this one's position in the map
// - Maintains a clear structure since the user always interaces with DataHandle objects
template <typename T>
DataHandle<T> DataManager::create_variable_kernel_handle(const DataUsage &buffer_usage, const MemoryHint &mem_hint,
                                                         size_t byte_size) {
    DataEntry entry;
    DataHandle<T> data_handle;

    entry.mem_hint = mem_hint;
    entry.data_usage = buffer_usage;
    entry.byte_size = byte_size;

    data_handle.ID = id_counter++;

    if (mem_hint == MemoryHint::DeviceLocal) {
        device_local_tasks_.push_back(entry);
    }

    data_map[data_handle.ID] = entry;

    return data_handle;
}

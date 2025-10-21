#include "DataManager.h"
#include "IGPUExecutor.h"
#include "TypeTraits.h"
#include <cstddef>
#include <memory>
#include <span>

template <typename T> DataHandle<T> DataManager::create_data_handle(T &data, const MemoryHint &mem_hint) {
    DataEntry entry;
    DataHandle<T> data_handle;

    auto data_ptr = std::make_shared<T>(std::move(data));

    entry.data = data_ptr;
    entry.mem_hint = mem_hint;

    // If true then T is a container type, else it's size can be found through sizeof() at compile time
    if constexpr (isContiguousContainer<T>::value) {
        size_t byte_size = data_ptr->size() * sizeof(typename T::value_type);
        entry.byte_size = byte_size;
        entry.raw_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(data_ptr->data()), byte_size);
        };
    } else {
        size_t byte_size = sizeof(T);
        entry.byte_size = byte_size;
        entry.raw_data_accessor = [data_ptr, byte_size]() {
            return std::span<const std::byte>(reinterpret_cast<const std::byte *>(std::addressof(*data_ptr)),
                                              byte_size);
        };
    }

    data_handle.ID = id_counter++;
    data_map[data_handle.ID] = entry;

    return data_handle;
}

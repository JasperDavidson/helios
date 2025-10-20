#include "DataManager.h"
#include <any>

template <typename T> DataHandle<T> DataManager::create_data_handle(T data) {
    DataEntry entry;
    DataHandle<T> data_handle;

    entry.byte_size = sizeof(data);
    entry.data = std::make_any<T>(data);

    data_handle.ID = id_counter++;
    data_map

        return data_handle;
}

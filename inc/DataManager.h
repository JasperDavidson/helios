#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

#include "IGPUExecutor.h"
#include <any>
#include <unordered_map>

// BaseDataHandle provides the ID needed for the runtime map
/*
class BaseDataHandle {
  public:
    int ID;

    BaseDataHandle(int ID) : ID(ID) {};
    virtual ~BaseDataHandle() = default;
};
*/

// Templated DataHandle allows for associating types needed for std::any casts
template <typename T> class DataHandle {
  public:
    int ID;
    DataHandle(int ID) : ID(ID) {};
};

struct DataEntry {
    std::any data;
    size_t byte_size;
    MemoryHint mem_hint;

    // Generic getter for accessing raw memory
    std::function<std::span<const std::byte>()> raw_data_accessor;
};

// DataManager object allows for caching of DataHandles to their actual objects
class DataManager {
  public:
    template <typename T> T get_data(DataHandle<T> data_handle) const {
        return std::any_cast<T>(data_map.at(data_handle.ID).data);
    }
    template <typename T> DataHandle<T> create_data_handle(T &&data, const MemoryHint &mem_hint);
    void store_data(int data_id, std::any new_data) { data_map.at(data_id).data = new_data; };

    std::span<const std::byte> get_data_span(int ID) const { return data_map.at(ID).raw_data_accessor(); };
    int get_data_length(int ID) const { return data_map.at(ID).byte_size; };
    MemoryHint get_mem_hint(int ID) const { return data_map.at(ID).mem_hint; };

  private:
    std::unordered_map<int, DataEntry> data_map;
    size_t id_counter = 0;
};

#endif

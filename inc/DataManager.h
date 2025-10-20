#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

#include <any>
#include <unordered_map>

// BaseDataHandle provides the ID needed for the runtime map
class BaseDataHandle {
  public:
    int ID;

    BaseDataHandle(int ID) : ID(ID) {};
    virtual ~BaseDataHandle() = default;
};

// Templated DataHandle allows for associating types needed for std::any casts
template <typename T> class DataHandle : public BaseDataHandle {
  public:
    DataHandle(int ID) : BaseDataHandle(ID) {};
};

struct DataEntry {
    std::any data;
    size_t byte_size;
};

// DataManager object allows for caching of DataHandles to their actual objects
class DataManager {
  public:
    template <typename T> T get_data(DataHandle<T> data_handle) { return data_map[data_handle.ID]; }
    template <typename T> DataHandle<T> create_data_handle(T data);

    int get_data_length(int ID) { return data_map[ID].byte_size; };

  private:
    std::unordered_map<int, DataEntry> data_map;
    size_t id_counter = 0;
};

#endif

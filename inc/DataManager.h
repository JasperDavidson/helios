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

// DataManager object allows for caching of DataHandles to their actual objects
class DataManager {
  public:
    template <typename T> T get_data(DataHandle<T> data_handle) { return data_manager[data_handle.ID]; }

  private:
    std::unordered_map<int, std::any> data_manager;
};

#endif

#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H

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

#endif DATA_HANDLE_H

#ifndef RUNTIME_H
#define RUNTIME_H

#include "DataHandle.h"
#include <any>
#include <unordered_map>

class Runtime {
  public:
    template <typename T> T get_data(DataHandle<T>);

  private:
    std::unordered_map<int, std::any> data_manager;
};

#endif

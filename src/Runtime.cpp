#include "Runtime.h"
#include "Tasks.h"
#include <any>

template <typename T> T Runtime::get_data(DataHandle<T> data_handle) {
    return std::any_cast<T>(data_manager[data_handle.ID]);
}

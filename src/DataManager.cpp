#include "DataManager.h"
#include "IGPUExecutor.h"
#include "TypeTraits.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>

std::span<std::byte> DataManager::get_span_mut(int data_id) {
    if (data_map.at(data_id).data_usage != DataUsage::ReadWrite) {
        throw std::runtime_error("Attempted to fetch mutable span into read-only data");
    }

    return data_map.at(data_id).raw_data_accessor();
};

#include "DataManager.h"
#include "TypeTraits.h"
#include <cstddef>
#include <iostream>
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

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

uint64_t next_pow2(uint64_t x) { return x <= 1 ? 1 : 1 << (64 - __builtin_clz(x - 1)); }

GPUMemoryAllocator::GPUMemoryAllocator(size_t devloc_min_size, size_t devloc_max_size, size_t hostvis_min_size,
                                       size_t hostvis_max_size, IGPUExecutor &device) {
    devloc_min_order = next_pow2(devloc_min_size);
    devloc_max_order = next_pow2(devloc_max_size);
    hostvis_min_order = next_pow2(hostvis_min_size);
    hostvis_max_order = next_pow2(hostvis_max_size);

    devloc_free_mask = 1 << devloc_max_order;
    hostvis_free_mask = 1 << hostvis_max_order;

    // Maybe not 0 here? Figure out what the mem offset is
    devloc_free_map[0] = true;
};

size_t GPUMemoryAllocator::allocate_memory(size_t mem_size, MemoryHint mem_hint) {
    // Find the minimum order (power of 2 memory size block) where this can be stored
    size_t order = next_pow2(mem_size);

    // Fetch the relevant instance variables based on memory hint
    std::vector<std::vector<size_t>> &size_address =
        mem_hint == MemoryHint::DeviceLocal ? devloc_size_address : hostvis_size_address;
    uint64_t &free_mask = mem_hint == MemoryHint::DeviceLocal ? devloc_free_mask : hostvis_free_mask;
    std::unordered_map<size_t, size_t> &free_map =
        mem_hint == MemoryHint::DeviceLocal ? devloc_free_map : hostvis_free_map;

    uint64_t search_mask = free_mask & ~((1 << order) - 1);
    if (search_mask == 0) {
        // Find some way to more effectively handle being out of memory
        // Some sort of futures system or multithreading?
        throw std::runtime_error("No space left on GPU!");
    }

    int next_free_order = __builtin_ctz(search_mask) + 1;
    size_t next_free_addr = size_address[next_free_order][-1];

    // Return if free block already available
    if (next_free_order == order) {
        size_address[next_free_order].pop_back();
        return next_free_addr;
    }

    // Break down the buddy addresses to provide space
    for (int cur_order = next_free_order; cur_order > order; cur_order--) {
        size_address[next_free_order].pop_back();

        size_t right_addr = next_free_addr + (1 << (cur_order - 1));
        size_address[cur_order - 1].push_back(right_addr);
        size_address[cur_order - 1].push_back(next_free_addr);

        free_map[right_addr] = size_address[cur_order - 1].size() - 2;
        free_map[next_free_addr] = size_address[cur_order - 1].size() - 1;
    }

    // Free the newly created split at next_free_addr
    size_address[order].pop_back();

    return next_free_addr;
}

void GPUMemoryAllocator::check_free_mem(size_t mem_size, size_t offset, MemoryHint mem_hint) {
    size_t order = next_pow2(mem_size);

    std::vector<std::vector<size_t>> &size_address =
        mem_hint == MemoryHint::DeviceLocal ? devloc_size_address : hostvis_size_address;
    uint64_t &free_mask = mem_hint == MemoryHint::DeviceLocal ? devloc_free_mask : hostvis_free_mask;
    size_t max_order = mem_hint == MemoryHint::DeviceLocal ? devloc_max_order : hostvis_max_order;
    std::unordered_map<size_t, size_t> &free_map =
        mem_hint == MemoryHint::DeviceLocal ? devloc_free_map : hostvis_free_map;

    // Merge back up the buddy addresses, if possible
    size_t cur_order = order;
    size_t new_free_addr = offset;
    size_t buddy_address = new_free_addr ^ (1 << order);
    for (int cur_order = order; cur_order < order; ++cur_order) {
        if (free_map.find(buddy_address) == free_map.end()) {
            break;
        }

        // Overwrite the buddy element with the last element in the order list for O(1) removal
        int buddy_order_idx = free_map[buddy_address];
        size_address[cur_order][buddy_order_idx] = size_address[cur_order][-1];
        size_address[cur_order].pop_back();
        free_map[size_address[cur_order][buddy_order_idx]] = buddy_order_idx;

        // Add to order list above
        // We want to push back the address at the order above that encapsulates both blocks -> check if already
        // aligned, else align
        size_t new_free_addr =
            new_free_addr % (2 ^ (cur_order + 1)) == 0 ? new_free_addr : (new_free_addr - (1 << cur_order));
        size_address[cur_order + 1].push_back(new_free_addr);

        buddy_address = new_free_addr ^ (1 << cur_order);
    }
}

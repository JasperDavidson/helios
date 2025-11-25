#include "IGPUExecutor.h"
#include "DataManager.h"
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

uint64_t next_pow2(uint64_t x) { return x <= 1 ? 1 : 1 << (64 - __builtin_clz(x - 1)); }

IGPUExecutor::GPUMemoryAllocator::GPUMemoryAllocator(size_t devloc_min_size, size_t devloc_max_size,
                                                     size_t unified_min_size, size_t unified_max_size,
                                                     size_t hostvis_min_size, size_t hostvis_max_size) {
    devloc_min_order = next_pow2(devloc_min_size);
    devloc_max_order = next_pow2(devloc_max_size);
    unified_min_order = next_pow2(unified_min_size);
    unified_max_order = next_pow2(unified_max_size);
    hostvis_min_order = next_pow2(hostvis_min_size);
    hostvis_max_order = next_pow2(hostvis_max_size);

    devloc_free_mask = 1 << devloc_max_order;
    unified_free_mask = 1 << unified_max_order;
    hostvis_free_mask = 1 << hostvis_max_order;

    // Maybe not 0 here? Figure out what the mem offset is
    devloc_free_map[0] = true;
    unified_free_map[0] = true;
    hostvis_free_map[0] = true;
};

IGPUExecutor::GPUMemoryAllocator::GPUMemoryAllocator() : GPUMemoryAllocator(0, 0, 0, 0, 0, 0) {};

void IGPUExecutor::GPUMemoryAllocator::init_mem_types(uint64_t &free_mask,
                                                      std::vector<std::vector<size_t>> &size_address,
                                                      std::unordered_map<size_t, size_t>, MemoryHint mem_hint) {
    // Fetch the relevant instance variables based on memory hint
    switch (mem_hint) {
    case MemoryHint::Unified:
        free_mask = unified_free_mask;
        size_address = unified_size_address;
        break;
    case MemoryHint::HostVisible:
        free_mask = hostvis_free_mask;
        size_address = hostvis_size_address;
        break;
    case MemoryHint::DeviceLocal:
        free_mask = devloc_free_mask;
        size_address = devloc_size_address;
        break;
    default:
        // Could possibly throw this on a per system level as well? (e.g. some systems won't have unified)
        throw std::runtime_error("Tried to allocate memory of invalid type");
    }
}

size_t IGPUExecutor::GPUMemoryAllocator::allocate_memory(size_t mem_size, MemoryHint mem_hint) {
    // Find the minimum order (power of 2 memory size block) where this can be stored
    size_t order = next_pow2(mem_size);

    // Fetch the relevant instance variables based on memory hint
    uint64_t &free_mask = devloc_free_mask;
    std::vector<std::vector<size_t>> &size_address = devloc_size_address;
    std::unordered_map<size_t, size_t> free_map = devloc_free_map;
    init_mem_types(free_mask, size_address, free_map, mem_hint);

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

void IGPUExecutor::GPUMemoryAllocator::check_free_mem(size_t mem_size, size_t offset, MemoryHint mem_hint) {
    size_t order = next_pow2(mem_size);

    // Fetch the relevant instance variables based on memory hint
    uint64_t &free_mask = devloc_free_mask;
    std::vector<std::vector<size_t>> &size_address = devloc_size_address;
    std::unordered_map<size_t, size_t> free_map = devloc_free_map;
    init_mem_types(free_mask, size_address, free_map, mem_hint);

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

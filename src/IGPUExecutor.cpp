#include "IGPUExecutor.h"
#include "DataManager.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

IGPUExecutor::GPUMemoryAllocator::GPUMemoryAllocator(size_t devloc_min_size, size_t devloc_max_size,
                                                     size_t unified_min_size, size_t unified_max_size,
                                                     size_t hostvis_min_size, size_t hostvis_max_size) {
    devloc_min_order = (int)log2(next_pow2(devloc_min_size));
    devloc_max_order = (int)log2(next_pow2(devloc_max_size));
    unified_min_order = (int)log2(next_pow2(unified_min_size));
    unified_max_order = (int)log2(next_pow2(unified_max_size));
    hostvis_min_order = (int)log2(next_pow2(hostvis_min_size));
    hostvis_max_order = (int)log2(next_pow2(hostvis_max_size));

    devloc_free_mask = (1ULL << devloc_max_order);
    unified_free_mask = (1ULL << unified_max_order);
    hostvis_free_mask = (1ULL << hostvis_max_order);

    devloc_size_address[devloc_max_order].push_back(0);
    devloc_free_map[devloc_max_order][0] = 0;

    unified_size_address[unified_max_order].push_back(0);
    unified_free_map[unified_max_order][0] = 0;

    hostvis_size_address[hostvis_max_order].push_back(0);
    hostvis_free_map[hostvis_max_order][0] = 0;
};

IGPUExecutor::GPUMemoryAllocator::GPUMemoryAllocator() : GPUMemoryAllocator(0, 0, 0, 0, 0, 0) {};

void IGPUExecutor::GPUMemoryAllocator::init_mem_types(
    uint64_t *&free_mask, std::unordered_map<uint8_t, std::vector<size_t>> *&size_address,
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> *&free_map, MemoryHint mem_hint) {
    // Fetch the relevant instance variables based on memory hint
    switch (mem_hint) {
    case MemoryHint::Unified:
        if (unified_max_order == 0) {
            throw std::runtime_error("Tried to allocate memory of invalid type: Unified");
        }

        free_mask = &unified_free_mask;
        size_address = &unified_size_address;
        free_map = &unified_free_map;
        break;
    case MemoryHint::HostVisible:
        if (hostvis_max_order == 0) {
            throw std::runtime_error("Tried to allocate memory of invalid type: HostVisible");
        }

        free_mask = &hostvis_free_mask;
        size_address = &hostvis_size_address;
        free_map = &hostvis_free_map;
        break;
    case MemoryHint::DeviceLocal:
        if (devloc_max_order == 0) {
            throw std::runtime_error("Tried to allocate memory of invalid type: DeviceLocal");
        }

        free_mask = &devloc_free_mask;
        size_address = &devloc_size_address;
        free_map = &devloc_free_map;
        break;
    }
}

size_t IGPUExecutor::GPUMemoryAllocator::allocate_memory(size_t mem_size, MemoryHint mem_hint) {
    // Find the minimum order (power of 2 memory size block) where this can be stored
    size_t order = (int)log2(next_pow2(mem_size));

    // Fetch the relevant instance variables based on memory hint
    uint64_t *free_mask = nullptr;
    std::unordered_map<uint8_t, std::vector<size_t>> *size_address = nullptr;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> *free_map = nullptr;
    init_mem_types(free_mask, size_address, free_map, mem_hint);

    size_t min_order = 0;
    switch (mem_hint) {
    case MemoryHint::DeviceLocal:
        min_order = devloc_min_order;
        break;
    case MemoryHint::Unified:
        min_order = unified_min_order;
        break;
    case MemoryHint::HostVisible:
        min_order = hostvis_min_order;
        break;
    default:
        throw std::runtime_error("Tried to work with memory of invalid type");
    }
    min_order = std::max(min_order, order);

    uint64_t search_mask = (*free_mask) & ~((1ULL << order) - 1);
    // std::cout << "Free mask before alloc: " << std::bitset<64>(*free_mask) << std::endl;
    //    std::cout << "Search mask: " << std::bitset<64>(search_mask) << std::endl;
    if (search_mask == 0) {
        // Find some way to more effectively handle being out of memory
        // Some sort of futures system or multithreading?
        throw std::runtime_error("No space available on GPU for block size!");
    }

    int next_free_order = __builtin_ctzll(search_mask);
    size_t next_free_addr = (*size_address)[next_free_order].back();

    // Free the newly created split at next_free_addr
    //    std::cout << "size addr size before: " << (*size_address)[next_free_order].size() << std::endl;
    //    std::cout << "free map size before: " << (*free_map)[next_free_order].size() << std::endl;
    (*size_address)[next_free_order].pop_back();
    (*free_map)[next_free_order].erase(next_free_addr);

    // Unmark the previous order as free if necessary
    if ((*size_address)[next_free_order].empty()) {
        (*free_mask) &= (~(1ULL << next_free_order));
    }

    // Break down the buddy addresses to provide space
    //   std::cout << "next free order: " << next_free_order << std::endl;
    //    std::cout << "end order: " << order << std::endl;
    for (int cur_order = next_free_order; min_order < cur_order; cur_order--) {
        int split_order = cur_order - 1;

        // Add both blocks to their respective free lists
        size_t right_addr = next_free_addr + (1ULL << (split_order));
        (*size_address)[split_order].push_back(right_addr);

        // Assign the right address be free in the free map
        (*free_map)[split_order][right_addr] = (*size_address)[split_order].size() - 1;

        // Mark the current order as containing a free block
        (*free_mask) |= (1ULL << (split_order));
    }

    //  std::cout << "Free mask after alloc: " << std::bitset<64>(*free_mask) << std::endl;
    //    std::cout << "size addr size after: " << (*size_address)[next_free_order].size() << std::endl;
    //    std::cout << "free map size after: " << (*free_map)[next_free_order].size() << std::endl;
    return next_free_addr;
}

void IGPUExecutor::GPUMemoryAllocator::check_free_mem(size_t mem_size, size_t offset, MemoryHint mem_hint) {
    // Fetch the relevant instance variables based on memory hint
    uint64_t *free_mask = nullptr;
    std::unordered_map<uint8_t, std::vector<size_t>> *size_address = nullptr;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> *free_map = nullptr;
    init_mem_types(free_mask, size_address, free_map, mem_hint);

    size_t max_order = 0;
    switch (mem_hint) {
    case MemoryHint::DeviceLocal:
        max_order = devloc_max_order;
        break;
    case MemoryHint::Unified:
        max_order = unified_max_order;
        break;
    case MemoryHint::HostVisible:
        max_order = hostvis_max_order;
        break;
    default:
        throw std::runtime_error("Tried to work with memory of invalid type");
    }

    // std::cout << "free mask before free: " << std::bitset<64>(*free_mask) << std::endl;

    size_t new_free_addr = offset;
    size_t cur_order = (int)log2(next_pow2(mem_size));

    // Catch if a double free occurs
    if ((*free_map)[cur_order].find(new_free_addr) != (*free_map)[cur_order].end()) {
        throw std::runtime_error("CRITICAL: Attempted to free memory twice");
    }

    while (cur_order < max_order) {
        size_t buddy_address = new_free_addr ^ (1ULL << (cur_order));
        //   std::cout << "buddy address: " << buddy_address << std::endl;

        // Check if merging can begin
        if ((*free_map)[cur_order].find(buddy_address) == (*free_map)[cur_order].end()) {
            //     std::cout << "did not find buddy" << std::endl;
            break;
        }
        //  std::cout << "found buddy" << std::endl;

        // Remove buddy from list and map
        int buddy_order_idx = (*free_map)[cur_order][buddy_address];
        size_t last_elem = (*size_address)[cur_order].back();

        // Swap and pop buddy with back element
        (*size_address)[cur_order][buddy_order_idx] = last_elem;
        (*free_map)[cur_order][last_elem] = buddy_order_idx;
        (*size_address)[cur_order].pop_back();
        (*free_map)[cur_order].erase(buddy_address);

        // Prepare for next iteration
        if ((*size_address)[cur_order].empty()) {
            (*free_mask) &= (~(1ULL << cur_order));
        }
        if (buddy_address < new_free_addr) {
            new_free_addr = buddy_address;
        }

        cur_order++;
    }

    // Mark as free memory after merging complete
    (*size_address)[cur_order].push_back(new_free_addr);
    (*free_map)[cur_order][new_free_addr] = (*size_address)[cur_order].size() - 1;
    (*free_mask) |= (1ULL << cur_order);
    // std::cout << "free mask after free: " << std::bitset<64>(*free_mask) << std::endl;
}

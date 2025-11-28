#include "DataManager.h"
#include "IGPUExecutor.h"

#include <gtest/gtest.h>

constexpr size_t MIN_SIZE = 4;
constexpr size_t MAX_SIZE = 256;

class GPUMemoryAllocatorTest : public testing::Test {
  protected:
    IGPUExecutor::GPUMemoryAllocator mem_alloc =
        IGPUExecutor::GPUMemoryAllocator(MIN_SIZE, MAX_SIZE, MIN_SIZE, MAX_SIZE, MIN_SIZE, MAX_SIZE);
};

TEST_F(GPUMemoryAllocatorTest, FullAllocFreeMin) {
    size_t min_offset = mem_alloc.allocate_memory(MIN_SIZE, MemoryHint::Unified);
    EXPECT_EQ(MAX_SIZE - MIN_SIZE, mem_alloc.unified_free_mask);

    mem_alloc.check_free_mem(MIN_SIZE, min_offset, MemoryHint::Unified);
    EXPECT_EQ(MAX_SIZE, mem_alloc.unified_free_mask);
}

TEST_F(GPUMemoryAllocatorTest, BuddyAllocationContiguous) {
    int middle_order =
        mem_alloc.unified_min_order + (int)(mem_alloc.unified_max_order - mem_alloc.unified_min_order) / 2;
    size_t block_offset = mem_alloc.allocate_memory(pow(2, middle_order), MemoryHint::Unified);
    std::cout << middle_order << '\n';

    for (int cur_order = middle_order + 1; cur_order < mem_alloc.unified_max_order; ++cur_order) {
        EXPECT_EQ(1, mem_alloc.unified_size_address[cur_order].size());
        EXPECT_EQ(1, mem_alloc.unified_free_map[cur_order].size());
    }

    size_t buddy_offset = mem_alloc.allocate_memory(pow(2, middle_order), MemoryHint::Unified);
    EXPECT_EQ(buddy_offset, block_offset + pow(2, middle_order));

    EXPECT_EQ(middle_order + 1, __builtin_ctzll(mem_alloc.unified_free_mask));
}

// Regression tests
// TEST_F(GPUMemoryAllocatorTest, BuddyRegressionTest)

// Out of order tests

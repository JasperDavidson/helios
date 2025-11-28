#include "DataManager.h"
#include "IGPUExecutor.h"

#include "gmock/gmock.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace testing;

class GPUMemoryAllocatorTest : public testing::Test {
  protected:
    size_t min_devloc_size = 4;
    size_t max_devloc_size = 256;
    size_t min_unified_size = 4;
    size_t max_unified_size = 256;
    size_t min_hostivs_size = 4;
    size_t max_hostvis_size = 256;

    void set_sizes(size_t min_devloc_size, size_t max_devloc_size, size_t min_unified_size, size_t max_unified_size,
                   size_t min_hostivs_size, size_t max_hostvis_size) {
        mem_alloc = IGPUExecutor::GPUMemoryAllocator(min_devloc_size, max_devloc_size, min_unified_size,
                                                     max_unified_size, min_hostivs_size, max_hostvis_size);

        min_devloc_size = 4;
        max_devloc_size = 256;
        min_unified_size = 4;
        max_unified_size = 256;
        min_hostivs_size = 4;
        max_hostvis_size = 256;
    }

    IGPUExecutor::GPUMemoryAllocator mem_alloc = IGPUExecutor::GPUMemoryAllocator(
        min_devloc_size, max_devloc_size, min_unified_size, max_unified_size, min_hostivs_size, max_hostvis_size);
};

// ****************************
// Bounds Tests
// ****************************

// Tests allocating blocks below minimum size
TEST_F(GPUMemoryAllocatorTest, AllocBelowMin) {
    size_t below_min_offset = mem_alloc.allocate_memory(pow(2, mem_alloc.unified_min_order - 1), MemoryHint::Unified);

    uint64_t expected_mask = max_unified_size - min_unified_size;
    EXPECT_EQ(expected_mask, mem_alloc.unified_free_mask);
}

// Tests allocating blocks above maximum size
TEST_F(GPUMemoryAllocatorTest, AllocAboveMax) {
    EXPECT_THROW(
        {
            try {
                mem_alloc.allocate_memory(
                    std::pow(pow(2, mem_alloc.unified_max_order) + 1, mem_alloc.unified_min_order - 1),
                    MemoryHint::Unified);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("No space available on GPU for block size!", e.what());
                throw;
            }
        },
        std::runtime_error);
}

// Tests allocating memory of invalid type
TEST_F(GPUMemoryAllocatorTest, AllocInvalid) {
    size_t devloc_offset;
    size_t hostvis_offset;
    size_t unified_offset;

    size_t test_size = 64;
    size_t test_min = 4;
    size_t test_max = 256;

    // Test all regions can be marked invalid while the others work

    set_sizes(test_min, test_max, 0, 0, test_min, test_max);
    devloc_offset = mem_alloc.allocate_memory(test_size, MemoryHint::DeviceLocal);
    hostvis_offset = mem_alloc.allocate_memory(test_size, MemoryHint::HostVisible);

    EXPECT_THROW(
        {
            try {
                mem_alloc.allocate_memory(test_size, MemoryHint::Unified);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Tried to allocate memory of invalid type: Unified", e.what());
                throw;
            }
        },
        std::runtime_error);

    mem_alloc.check_free_mem(test_size, devloc_offset, MemoryHint::DeviceLocal);
    mem_alloc.check_free_mem(test_size, hostvis_offset, MemoryHint::HostVisible);

    set_sizes(0, 0, test_min, test_max, test_min, test_max);
    unified_offset = mem_alloc.allocate_memory(test_size, MemoryHint::Unified);
    hostvis_offset = mem_alloc.allocate_memory(test_size, MemoryHint::HostVisible);

    EXPECT_THROW(
        {
            try {
                mem_alloc.allocate_memory(test_size, MemoryHint::DeviceLocal);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Tried to allocate memory of invalid type: DeviceLocal", e.what());
                throw;
            }
        },
        std::runtime_error);

    mem_alloc.check_free_mem(test_size, unified_offset, MemoryHint::Unified);
    mem_alloc.check_free_mem(test_size, hostvis_offset, MemoryHint::HostVisible);

    set_sizes(test_min, test_max, test_min, test_max, 0, 0);
    unified_offset = mem_alloc.allocate_memory(test_size, MemoryHint::Unified);
    devloc_offset = mem_alloc.allocate_memory(test_size, MemoryHint::DeviceLocal);

    EXPECT_THROW(
        {
            try {
                mem_alloc.allocate_memory(test_size, MemoryHint::HostVisible);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Tried to allocate memory of invalid type: HostVisible", e.what());
                throw;
            }
        },
        std::runtime_error);
}

// ****************************
// Basic Alloc and Free Tests
// ****************************

// Test correct updating for allocating down to the lowest memory block size
TEST_F(GPUMemoryAllocatorTest, FullAllocFreeMin) {
    size_t min_offset = mem_alloc.allocate_memory(min_unified_size, MemoryHint::Unified);
    uint64_t expected_mask = max_unified_size - min_unified_size;
    EXPECT_EQ(expected_mask, mem_alloc.unified_free_mask);

    // Test each block size has exactly one option free
    for (int cur_order = mem_alloc.unified_max_order - 1; cur_order >= mem_alloc.unified_min_order; --cur_order) {
        EXPECT_EQ(1, mem_alloc.unified_size_address[cur_order].size());
        EXPECT_EQ(1, mem_alloc.unified_free_map[cur_order].size());
    }

    // Test that all memory gets merged after free of only block
    mem_alloc.check_free_mem(min_unified_size, min_offset, MemoryHint::Unified);
    EXPECT_EQ(max_unified_size, mem_alloc.unified_free_mask);
    EXPECT_EQ(1, mem_alloc.unified_size_address[mem_alloc.unified_max_order].size());
    EXPECT_EQ(1, mem_alloc.unified_free_map[mem_alloc.unified_max_order].size());
}

// Check that buddies (and memory generally when uniform) allocate contiguously
TEST_F(GPUMemoryAllocatorTest, BuddyAllocationContiguous) {
    int middle_order =
        mem_alloc.unified_min_order + (int)(mem_alloc.unified_max_order - mem_alloc.unified_min_order) / 2;
    int middle_size = pow(2, middle_order);

    // Test that free sizes appropriately change as same-size memory is allocated
    size_t block_offset = mem_alloc.allocate_memory(middle_size, MemoryHint::Unified);
    for (int cur_order = middle_order + 1; cur_order < mem_alloc.unified_max_order; ++cur_order) {
        EXPECT_EQ(1, mem_alloc.unified_size_address[cur_order].size());
        EXPECT_EQ(1, mem_alloc.unified_free_map[cur_order].size());
    }

    size_t buddy_offset = mem_alloc.allocate_memory(middle_size, MemoryHint::Unified);
    EXPECT_EQ(buddy_offset, block_offset + pow(2, middle_order));
    for (int cur_order = middle_order; cur_order > mem_alloc.unified_min_order; --cur_order) {
        EXPECT_EQ(0, mem_alloc.unified_size_address[cur_order].size());
        EXPECT_EQ(0, mem_alloc.unified_free_map[cur_order].size());
    }

    size_t outlier_offset = mem_alloc.allocate_memory(middle_size, MemoryHint::Unified);
    EXPECT_EQ(outlier_offset, buddy_offset + pow(2, middle_order));
    EXPECT_EQ(0, mem_alloc.unified_size_address[middle_order + 1].size());
    EXPECT_EQ(0, mem_alloc.unified_free_map[middle_order + 1].size());

    // Test that all memory gets merged after free of all blocks
    mem_alloc.check_free_mem(middle_size, block_offset, MemoryHint::Unified);
    mem_alloc.check_free_mem(middle_size, outlier_offset, MemoryHint::Unified);
    mem_alloc.check_free_mem(middle_size, buddy_offset, MemoryHint::Unified);

    EXPECT_EQ(max_unified_size, mem_alloc.unified_free_mask);
    EXPECT_EQ(1, mem_alloc.unified_size_address[mem_alloc.unified_max_order].size());
    EXPECT_EQ(1, mem_alloc.unified_free_map[mem_alloc.unified_max_order].size());
}

// Allocate min then allocate buddy in middle
TEST_F(GPUMemoryAllocatorTest, AllocMiddleAfterMin) {}

// Allocate across full range

// Regression tests
// TEST_F(GPUMemoryAllocatorTest, BuddyRegressionTest)

// Out of order tests

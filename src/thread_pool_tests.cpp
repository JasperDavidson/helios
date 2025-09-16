#include "ThreadPool.h"
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

void flip(std::atomic<bool> &atomic_bool) {
  atomic_bool.exchange(!atomic_bool.load());
}

int add(int a, int b) { return a + b; }

class ThreadPoolTest : public testing::Test {
protected:
  std::unique_ptr<ThreadPool> single_thread_pool_;

  std::atomic<bool> atomic_bool = false;

  void SetUp() override {
    single_thread_pool_ = std::make_unique<ThreadPool>(1);
  }
};

// Creating a pool with zero threads should thrown an out of range exception
TEST_F(ThreadPoolTest, ZeroThreads) {
  EXPECT_THROW(ThreadPool(0), std::out_of_range);
}

// A single task to flip a booleans value should succeed with one thread in the
// pool
TEST_F(ThreadPoolTest, SingleThreadSingleTask) {
  bool prev_bool = atomic_bool.load();
  single_thread_pool_->add_task(flip, std::ref(atomic_bool)).get();

  ASSERT_NE(prev_bool, atomic_bool.load());
}

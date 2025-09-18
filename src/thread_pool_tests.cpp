#include "ThreadPool.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <thread>

const size_t SMALL_POOL_SIZE = 5;

void flip_test(std::atomic<bool> &atomic_bool) {
  atomic_bool.exchange(!atomic_bool.load());
}

void add_test(std::atomic<int> &a, std::atomic<int> &b,
              std::atomic<int> &result) {
  result = (a.load() + b.load());
}

void slp_test(int num_milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(num_milliseconds));
}

class ThreadPoolTest : public testing::Test {
protected:
  std::shared_ptr<ThreadPool> single_thread_pool_;
  std::shared_ptr<ThreadPool> small_thread_pool_;

  std::atomic<bool> atomic_bool = false;
  std::atomic<int> atomic_int_1 = 12;
  std::atomic<int> atomic_int_2 = 4;

  void SetUp() override {
    single_thread_pool_ = std::make_unique<ThreadPool>(1);
    small_thread_pool_ = std::make_unique<ThreadPool>(5);
  }
};

// Creating a pool with zero threads should thrown an out of range exception
TEST_F(ThreadPoolTest, ZeroThreads) {
  EXPECT_THROW(ThreadPool(0), std::out_of_range);
}

// Assign a pool with one thread one task
TEST_F(ThreadPoolTest, SingleThreadSingleTask) {
  bool prev_bool = atomic_bool.load();
  single_thread_pool_->add_task(flip_test, std::ref(atomic_bool)).get();

  ASSERT_NE(prev_bool, atomic_bool.load());
}

// Assign a pool with multiple threads one task
// Ensure that a single thread is assigned the task
TEST_F(ThreadPoolTest, MultipleThreadSingleTask) {
  int expected_add = atomic_int_1.load() + atomic_int_2.load();
  std::atomic<int> atomic_int_result = 0;

  small_thread_pool_
      ->add_task(add_test, std::ref(atomic_int_1), std::ref(atomic_int_2),
                 std::ref(atomic_int_result))
      .get();

  ASSERT_EQ(expected_add, atomic_int_result.load());
}

void test_NTasksMThreads_sleep(int sleep_time, int num_tasks, int num_threads) {
  ThreadPool thread_pool = ThreadPool(num_threads);

  // Sleep time * number of computations in parallel at a time + overhead
  // accounting
  int parallel_time =
      sleep_time * (int)std::ceil((double)num_tasks / (double)num_threads) +
      (num_tasks * num_threads);

  std::vector<std::future<void>> slp_futures;
  for (int i = 0; i < num_tasks; ++i) {
    slp_futures.push_back(thread_pool.add_task(slp_test, sleep_time));
  }

  auto parallel_start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_tasks; ++i) {
    slp_futures[i].get();
  }

  auto parallel_end_time = std::chrono::high_resolution_clock::now();
  auto parallel_duration = parallel_end_time - parallel_start_time;
  auto parallel_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(parallel_duration)
          .count();

  auto linear_start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_tasks; ++i) {
    slp_test(sleep_time);
  }

  auto linear_end_time = std::chrono::high_resolution_clock::now();
  auto linear_duration = linear_end_time - linear_start_time;
  auto linear_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(linear_duration)
          .count();

  std::cout << "Parallel: " << parallel_milliseconds << '\n';
  std::cout << "Linear: " << linear_milliseconds << '\n';

  ASSERT_LE(parallel_milliseconds, linear_milliseconds);
  ASSERT_LE(parallel_milliseconds, parallel_time);
}

// Assign a pool with n threads n tasks
// Ensure that a pool can run it's capacity worth of threads at once
TEST_F(ThreadPoolTest, NThreadsNTasks) {
  int sleep_time = 100;
  test_NTasksMThreads_sleep(sleep_time, SMALL_POOL_SIZE, SMALL_POOL_SIZE);
}

// Assign a pool with n threads 5n tasks
// Ensures that a pool efficiently runs 5n tasks of the same thing (as compared
// to linearly)
TEST_F(ThreadPoolTest, NThreadsGreaterTasks) {
  int sleep_time = 100;
  test_NTasksMThreads_sleep(sleep_time, SMALL_POOL_SIZE * 5, SMALL_POOL_SIZE);
}

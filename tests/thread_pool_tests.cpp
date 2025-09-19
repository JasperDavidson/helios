#include "ThreadPool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
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

void throw_exception_test() { throw std::runtime_error("Test exception!"); }

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

  ASSERT_LE(parallel_milliseconds, linear_milliseconds);
  ASSERT_LE(parallel_milliseconds, parallel_time);
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
// Ensures a single thread pool correctly executes a task
TEST_F(ThreadPoolTest, SingleThreadSingleTask) {
  bool prev_bool = atomic_bool.load();
  single_thread_pool_->add_task(flip_test, std::ref(atomic_bool)).get();

  ASSERT_NE(prev_bool, atomic_bool.load());
}

// Assign a pool with multiple threads one task
// Ensure that a single thread is assigned the task and it is correctly executed
TEST_F(ThreadPoolTest, MultipleThreadSingleTask) {
  int expected_add = atomic_int_1.load() + atomic_int_2.load();
  std::atomic<int> atomic_int_result = 0;

  small_thread_pool_
      ->add_task(add_test, std::ref(atomic_int_1), std::ref(atomic_int_2),
                 std::ref(atomic_int_result))
      .get();

  ASSERT_EQ(expected_add, atomic_int_result.load());
}

// Assign a pool with n threads n tasks
// Ensure that a pool can run it's capacity worth of threads at once
TEST_F(ThreadPoolTest, NThreadsNTasks) {
  int sleep_time = 50;
  test_NTasksMThreads_sleep(sleep_time, SMALL_POOL_SIZE, SMALL_POOL_SIZE);
}

// Assign a pool with n threads 5n tasks
// Ensures that a pool efficiently runs 5n tasks of the same thing (as compared
// to linearly)
TEST_F(ThreadPoolTest, NThreadsGreaterTasks) {
  int sleep_time = 50;
  test_NTasksMThreads_sleep(sleep_time, SMALL_POOL_SIZE * 3, SMALL_POOL_SIZE);
}

// Exception Safety Test
// A worker thread should catch any exception from a task and *not* die. It
// should instead be returned by the original std::future
TEST_F(ThreadPoolTest, ExceptSafety) {
  auto exception_future = small_thread_pool_->add_task(throw_exception_test);
  ASSERT_ANY_THROW(exception_future.get());
}

// Except Continue Test
// Verifies that if a batch of tasks are submitted to the thread pool and some
// throw exceptions, the rest still complete
TEST_F(ThreadPoolTest, ExceptContinue) {
  std::vector<std::future<void>> sleep_futures;
  int sleep_time = 50;
  int sleep_time_mult = 3;
  int expected_time_max =
      sleep_time_mult * sleep_time + (sleep_time_mult * SMALL_POOL_SIZE);

  for (int i = 0; i < (SMALL_POOL_SIZE * (sleep_time_mult - 1)) + 1; ++i) {
    sleep_futures.push_back(small_thread_pool_->add_task(slp_test, sleep_time));
    sleep_futures.push_back(small_thread_pool_->add_task(throw_exception_test));
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < sleep_futures.size(); ++i) {
    if (i % 2 != 0) {
      ASSERT_ANY_THROW(sleep_futures[i].get());
    } else {
      sleep_futures[i].get();
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = end_time - start_time;
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  ASSERT_LE(milliseconds, expected_time_max);
}

// TODO: Move-Only Types Test: Test with tasks that take ownership of, or
// return, move only types like std::unique_ptr
// Cases to consider:
// User tries to move a unique pointer into two different tasks and assign them
// to the thread pool, User inserts a task that returns a unique pointer

TEST_F(ThreadPoolTest, MoveOnlyTypeDifferentTasks) {}

TEST_F(ThreadPoolTest, MoveOnlyReturnType) {}

// TODO: Deadlock Test: Show that this basic implemenation will result in
// deadlock if, for a single thread pool, the submitted task submits a task to
// the same pool and the original task tries to fetch its result immediately -->
// Maybe implement work-stealing eventually?

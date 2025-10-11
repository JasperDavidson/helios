#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

class ThreadPool {
public:
  ThreadPool(size_t num_threads);
  ~ThreadPool();

  template <typename F, class... Types>
  auto add_task(F &&task, Types &&...task_args)
      -> std::future<std::invoke_result_t<F, Types...>> {
    // Wrap the task call with its arguments inside a lambda such that the queue
    // can always store consistent type of std::function<void()>

    // Using a packaged type + futures allows the outside user to access the
    // results of the task
    using TaskReturnType = std::invoke_result_t<F, Types...>;
    auto bound_task =
        [captured_task = std::forward<F>(task),
         captured_args = std::make_tuple(std::forward<Types>(task_args)...)]() {
          std::apply(captured_task, captured_args);
        };
    std::shared_ptr<std::packaged_task<TaskReturnType()>> task_package =
        std::make_shared<std::packaged_task<TaskReturnType()>>(bound_task);
    std::future<TaskReturnType> task_future = task_package->get_future();

    auto wrapper = [task_package]() { (*task_package)(); };

    {
      // Lock the mutex, emplace the task into the queue, then unlock the mutex
      // This prevents threads from accessing the queue as tasks are being added
      std::unique_lock<std::mutex> lock(queue_mtx_);
      task_queue_.emplace(wrapper);
    }

    // Notify just one thread (undeterministic) that the task queue is ready to
    // be read from
    cv_.notify_one();

    return task_future;
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> task_queue_;
  std::mutex queue_mtx_;
  std::condition_variable cv_;
  bool stop_ = false;

  void worker_loop();
};

#endif

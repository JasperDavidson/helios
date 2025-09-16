#include "ThreadPool.h"

#include <functional>
#include <mutex>
#include <stdexcept>

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads <= 0) {
    throw std::out_of_range(
        "The number of threads in the thread pool must be greater than 0");
  }

  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([this] { this->worker_loop(); });
  }
};

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> queue_lock(this->queue_mtx);
    this->stop = true;
  }

  this->cv.notify_all();

  for (std::thread &thread : this->workers) {
    thread.join();
  }
}

void ThreadPool::worker_loop() {
  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> queue_lock(this->queue_mtx);

      // Wait until task is ready or pool is being ended
      this->cv.wait(queue_lock,
                    [this] { return this->stop || !this->task_queue.empty(); });

      // If stopping and queue is empty -> return
      // Else, empty the task queue
      if (this->stop && this->task_queue.empty()) {
        return;
      }

      task = std::move(task_queue.front());
      this->task_queue.pop();
    }

    task();
  }
}

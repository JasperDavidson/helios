#include "ThreadPool.h"
#include <functional>

ThreadPool::ThreadPool(size_t num_threads) : num_threads(num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([this] {
      this->worker_loop();
    });
  }
};

void ThreadPool::add_task(std::function<void()> task) {
  {
    this->task_queue.emplace(task);
    // lock here?
  }
  this->cv.notify_one();
}

void ThreadPool::worker_loop() {
  while (true) {

  }
}

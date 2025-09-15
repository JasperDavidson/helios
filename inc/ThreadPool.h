#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class ThreadPool {
public:
  ThreadPool(size_t num_threads);
  ~ThreadPool();

  void add_task(std::function<void()> task);

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> task_queue;
  std::mutex mtx;
  std::condition_variable cv;
  bool stop = false;
  int num_threads;

  void worker_loop();
};

#endif

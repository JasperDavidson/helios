#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"
#include <condition_variable>
#include <memory>
#include <mutex>

class Scheduler {
  public:
    Scheduler(DataManager &data_manager, std::unique_ptr<ThreadPool> &thread_pool,
              std::unique_ptr<IGPUExecutor> &gpu_executor)
        : data_manager(data_manager), thread_pool(thread_pool), gpu_executor(gpu_executor) {};

    // TODO: For both visit methods, implement event polling -> wrap in a lambda that pushes to thread safe queue
    template <typename F, class... Types> void visit(const CPUTask<F, Types...> &cpu_task);
    void visit(const GPUTask &gpu_task);

    bool check_kernel_status(const std::string &kernel_name) { return gpu_executor->get_kernel_status(kernel_name); }

    void execute_graph(const TaskGraph &task_graph);

  private:
    class CompletionQueue {
      private:
        mutable std::mutex queue_mut;
        std::condition_variable cond_var;

      public:
        // Public so the scheduler can drain the queue
        std::queue<int> data_queue;

        // Locks the mutex -> pushes and notifies scheduler of completion
        void push_task(int task_id) {
            std::lock_guard<std::mutex> lock(queue_mut);
            data_queue.push(task_id);
            cond_var.notify_one();
        }

        // Waits for a task to notify the scheduler thread of it's completion
        std::unique_lock<std::mutex> wait() {
            std::unique_lock<std::mutex> lock(queue_mut);
            cond_var.wait(lock, [this] { return !data_queue.empty(); });

            return lock;
        }
    };

    CompletionQueue completed_queue;
    DataManager data_manager;
    std::unique_ptr<ThreadPool> &thread_pool;
    std::unique_ptr<IGPUExecutor> &gpu_executor;
};

#endif

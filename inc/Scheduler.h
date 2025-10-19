#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"

class Scheduler {
  public:
    template <typename F, class... Types> void visit(const CPUTask<F, Types...> &cpu_task) {
        thread_pool->add_task(cpu_task.task_lambda);
        // Also incorporate cpu task future in some sort of way as so scheduler can check when done?
    };
    void visit(const GPUTask &gpu_task);

  private:
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_executor;
};

#endif

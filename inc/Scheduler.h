#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"

class Scheduler {
  public:
    template <typename F, class... Types> void visit(const CPUTask<F, Types...> &cpu_task) const {
        thread_pool->add_task(cpu_task.task_lambda);
    };
    void visit(const GPUTask &gpu_task) const;

    template <typename F, class... Types> void execute_graph(const TaskGraph &task_graph);

  private:
    DataManager data_manager;
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_executor;
};

#endif

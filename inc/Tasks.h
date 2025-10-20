#ifndef TASK_H
#define TASK_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include <functional>
#include <future>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// Forward declaration of Scheduler class
class Scheduler;

class ITask {
  public:
    int ID;
    std::string task_name;
    std::vector<std::shared_ptr<BaseDataHandle>> inputs;
    std::vector<std::shared_ptr<BaseDataHandle>> outputs;

    ITask(int ID, const std::string &task_name, const std::vector<std::shared_ptr<BaseDataHandle>> &inputs,
          const std::vector<std::shared_ptr<BaseDataHandle>> &outputs)
        : ID(ID), task_name(task_name), inputs(inputs), outputs(outputs) {};
    ITask() = default;

    virtual void accept(const Scheduler &scheduler) = 0;
};

template <typename F, class... Types> class CPUTask : public ITask {
  public:
    std::function<void()> task_lambda;

    CPUTask(int ID, std::string task_name, const std::vector<std::shared_ptr<BaseDataHandle>> &inputs,
            const std::vector<std::shared_ptr<BaseDataHandle>> &outputs, const DataManager &data_manager, F &&task,
            Types &&...args)
        : ITask(ID, task_name, inputs, outputs) {
        // Three key things
        // 1. Work lambda to fetch data and run method
        // 2. Wrap that lambda in a packaged task that the scheduler can execute
        // 3. Include the packaged tasks' future so that the scheduler can tell when the task is done
        auto work_lambda = [=]() {
            auto tupled_handles = std::make_tuple(std::forward<Types>(args)...);
            std::apply([&](auto &&...handles) { task(data_manager.get_data(handles)...); }, tupled_handles);
        };

        using TaskReturnType = std::invoke_result<F, Types...>;
        std::shared_ptr<std::packaged_task<TaskReturnType>> task_package =
            std::make_shared<std::packaged_task<TaskReturnType>>(work_lambda);

        task_future = task_package->get_future();
        task_lambda = [task_package]() { (*task_package)(); };
    };

  private:
    std::future<std::invoke_result<F, Types...>> task_future;

    void accept(const Scheduler &scheduler) override;
};

class GPUTask : public ITask {
  public:
    GPUTask(int ID, const std::string &task_name, const std::vector<std::shared_ptr<BaseDataHandle>> &inputs,
            const std::vector<BufferUsage> &buffer_usages, const std::vector<std::shared_ptr<BaseDataHandle>> &outputs,
            const std::vector<int> &kernel_size, const std::vector<int> &threads_per_group)
        : ITask(ID, task_name, inputs, outputs), kernel_size(kernel_size), threads_per_group(threads_per_group),
          buffer_usages(buffer_usages) {};

    std::vector<int> kernel_size;
    std::vector<int> threads_per_group;
    std::vector<BufferUsage> buffer_usages;

  private:
    void accept(const Scheduler &scheduler) override;
};

class TaskGraph {
  private:
    // Adjacency list structure --> Graph should not be dense so this saves on memory
};

#endif

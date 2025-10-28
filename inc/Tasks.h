#ifndef TASK_H
#define TASK_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
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

    bool task_complete() const { return task_future.wait_for(0) == std::future_status::ready; };

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

/*
 * TaskGraph
 * Represent tasks (nodes) and their dependencies (edges)
 *
 *  So what should the task graph actually contain?
 *  Instance variables:
 *      - all_tasks_: map<int, shared_ptr<ITask>>
 *      - dependencies_: map<int (taskID), vector<int> (taskIDs)> - maps a task to all tasks that depend on it
 *      - dependent_: map<int, (taskID), vector<int> (taskIDs)> - maps a task to all tasks that it depends on
 *      - data_producer_map_: map<int (dataID), int (taskID)> - maps a data handle to the task that produces it
 *      - unfulfilled_data_: map<int (dataID), vector<int> (taskIDs)> - maps a data handle with no producer to a vector
 * tasks that require it
 * Methods:
 *      - add_task(std::shared_ptr<ITask> task): adds a task to the graph following this algorithm
 *          1. Add a shared pointer to the task to the all_tasks_ map
 *          2. Iterate over the output data handles and add data_producer_map_[handle->ID] = new_task->ID
 *              - What if a task already claimed to produce this data? (data_producer_map_.find(handle->ID) != .end())
 *              - Throw a runtime exception, since only one task con produce a certain piece of data
 *          3. Iterate over the input data handles and fetch their producer_task_ids.
 *              - If data_producer_map_[handle->ID] doesn't exist then add new_task->ID to unfulfilled_data_[handle->ID]
 *              - Add producer_task_id to list at dependencies_map_[new_task->ID]
 *              - Add new_task->ID to list at dependents_map_[producer_task_id]
 *              - If handle->ID is a key in unfulfilled_data_ iterate over the tasks and do
 *      - validate_graph()
 * depedencies[unfilfilled_task->ID] = new_task->ID
 */
class TaskGraph {
  public:
    TaskGraph();

    void add_task(std::shared_ptr<ITask> task);
    std::vector<int> find_ready() const;
    void validate_graph();

    std::vector<int> get_task_ids() const;
    std::shared_ptr<ITask> get_task(int task_id) const { return all_tasks_.at(task_id); };
    std::vector<int> get_dependents(int task_id) const { return dependents_.at(task_id); };
    std::vector<int> get_dependencies(int task_id) const { return dependencies_.at(task_id); };

  private:
    std::unordered_map<int, std::shared_ptr<ITask>> all_tasks_;
    std::unordered_map<int, std::vector<int>> dependencies_;
    std::unordered_map<int, std::vector<int>> dependents_;
    std::unordered_map<int, int> data_producer_map_;
    std::unordered_map<int, std::vector<int>> unfulfilled_data_;
};

#endif

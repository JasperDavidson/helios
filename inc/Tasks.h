#ifndef TASK_H
#define TASK_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration of Scheduler class
class Scheduler;

class ITask {
  public:
    int ID;
    std::string task_name;
    std::vector<int> input_ids;
    int output_id;

    ITask(int ID, const std::string &task_name, const std::vector<int> &input_ids, int output_id)
        : ID(ID), task_name(task_name), input_ids(input_ids), output_id(output_id) {};
    ITask() = default;

    virtual void accept(Scheduler &scheduler) = 0;
};

template <typename F, class... Types> class CPUTask : public ITask {
  public:
    std::function<void()> task_lambda;

    CPUTask(int ID, std::string task_name, const std::vector<int> &input_ids, int output_id, DataManager &data_manager,
            F &&task, Types &&...args)
        : ITask(ID, task_name, input_ids, output_id) {
        task_lambda = [=, task = std::forward<F>(task),
                       args_tuple = std::make_tuple(std::forward<Types>(args)...)]() mutable {
            // Bundle the input data
            auto inputs = std::apply(
                [&](auto &&...handles) { return std::make_tuple(data_manager.get_data(handles)...); }, args_tuple);

            // Compute the results
            auto result = std::apply(task, inputs);

            // Store the results in the output handle
            data_manager.store_data(output_id, result);
        };
    };

  private:
    void accept(Scheduler &scheduler) override;
};

class GPUTask : public ITask {
    // TODO: Need to generate a future for each task by analyzing the GPU somehow
    // We could encode a lambda that checks if the bool map of dispatch complete has been updated for this task
    //  - We can't actually check the map because the gpu executor is stored in the scheduler
    // Should we generate a future for every GPU task when the scheduler begins to execute the graph?
    //  - No, this would be inefficient + runtime type checking for tasks
    // Maybe the scheduler visitor design method can return/emplace a future since it *does* have access to the gpu?
    // (Check scheduler comments)
  public:
    GPUTask(int ID, const std::string &task_name, const std::vector<int> &input_ids,
            const std::vector<BufferUsage> &buffer_usages, int output_id, const std::vector<int> &kernel_size,
            const std::vector<int> &threads_per_group)
        : ITask(ID, task_name, input_ids, output_id), kernel_size(kernel_size), threads_per_group(threads_per_group),
          buffer_usages(buffer_usages) {};

    std::vector<int> kernel_size;
    std::vector<int> threads_per_group;
    std::vector<BufferUsage> buffer_usages;

  private:
    void accept(Scheduler &scheduler) override;
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

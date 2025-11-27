#ifndef TASK_H
#define TASK_H

#include "DataManager.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

const int ROOT_NODE_ID = -1;
const int VOID_RETURN = -1;

// Forward declaration of Scheduler class
class Scheduler;

class ITask {
  public:
    int id;
    std::string task_name;
    std::vector<int> input_ids;
    int output_id;

    ITask(const std::string &task_name, const std::vector<int> &input_ids, int output_id)
        : task_name(task_name), input_ids(input_ids), output_id(output_id) {};
    ITask() = default;

    virtual void accept(Scheduler &scheduler) = 0;
};

class BaseCPUTask : public ITask {
  public:
    BaseCPUTask(const std::string &task_name, const std::vector<int> &input_ids, int output_id)
        : ITask(task_name, input_ids, output_id) {};

    std::function<void()> task_lambda;
    void accept(Scheduler &scheduler) override;
};

template <typename F, class... Types> class TypedCPUTask : public BaseCPUTask {
  public:
    TypedCPUTask(std::string task_name, const std::vector<int> &input_ids, int output_id, DataManager &data_manager,
                 F &&task, Types &&...args)
        : BaseCPUTask(task_name, input_ids, output_id) {
        task_lambda = [&data_manager, output_id, task = std::forward<F>(task),
                       args_tuple = std::make_tuple(std::forward<Types>(args)...)]() mutable {
            // Bundle the input data
            auto inputs =
                std::apply([&](auto &&...handles) { return std::forward_as_tuple(data_manager.get_data(handles)...); },
                           args_tuple);

            // Store the results in the output handle if non-void return type
            using ReturnType = decltype(std::apply(task, inputs));
            if constexpr (std::is_void_v<ReturnType>) {
                std::apply(task, inputs);
            } else {
                auto result = std::apply(task, inputs);
                data_manager.store_data(output_id, result);
            }
        };
    };
};

template <typename F, class... Types>
TypedCPUTask(std::string, const std::vector<int> &, int, DataManager &, F &&, Types &&...)
    -> TypedCPUTask<std::decay_t<F>, Types...>;

class GPUTask : public ITask {
    // TODO: How should we actually capture the data from the GPU?
    //  - Implement a callback lambda that gets ran after the kernel is compute, transport memory back to output handle
    //  - Since output handle is a 'placeholder' create a 'real' DataHandle that replaces it once result is achieved
    //  KEY CHANGE: User must specify output size of kernel if the output size will differ from size of input objects,
    //  or opt into buffer counting
  public:
    GPUTask(const std::string &task_name, const std::vector<int> &input_ids, int output_id, bool count_buffer_active,
            int threads, const std::vector<int> &block_dim = {8, 8, 8})
        : ITask(task_name, input_ids, output_id), count_buffer_active(count_buffer_active), threads(threads),
          block_dim(block_dim) {};

    int threads;
    std::vector<int> block_dim;
    bool count_buffer_active;

  private:
    void accept(Scheduler &scheduler) override;
};

/*
 * TaskGraph
 * Represent tasks (nodes) and their dependencies (edges)
 * Contains a root node with initial inputs such that root tasks aren't treated as having unfulfilled dependencies
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
    TaskGraph() {};

    void add_task(std::shared_ptr<ITask> task, bool add_task);
    std::vector<int> find_ready() const;
    void validate_graph();

    std::vector<int> get_task_ids() const;
    std::shared_ptr<ITask> get_task(int task_id) const { return all_tasks_.at(task_id); };
    std::vector<int> get_dependents(int task_id) const {
        if (dependents_.find(task_id) != dependents_.end()) {
            return dependents_.at(task_id);
        }

        return {};
    };
    std::vector<int> get_dependencies(int task_id) const {
        if (dependencies_.find(task_id) != dependencies_.end()) {
            return dependencies_.at(task_id);
        }

        return {};
    };

  private:
    int task_id_inc = 0;

    std::unordered_map<int, std::shared_ptr<ITask>> all_tasks_;
    std::unordered_map<int, std::vector<int>> dependencies_;
    std::unordered_map<int, std::vector<int>> dependents_;
    std::unordered_map<int, int> data_producer_map_;
    std::unordered_map<int, std::vector<int>> unfulfilled_data_;
};

#endif

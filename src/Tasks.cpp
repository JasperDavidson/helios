#include "Tasks.h"
#include "Scheduler.h"
#include <algorithm>
#include <deque>
#include <stdexcept>
#include <unordered_map>

template <typename F, class... Types> void CPUTask<F, Types...>::accept(const Scheduler &scheduler) {
    scheduler.visit(*this);
}

void GPUTask::accept(const Scheduler &scheduler) { scheduler.visit(*this); }

void TaskGraph::add_task(std::shared_ptr<ITask> task) {
    all_tasks_[task->ID] = task;

    // Mapping each output of the task to the task itself
    for (auto &output_handle : task->outputs) {
        if (data_producer_map_.find(output_handle->ID) != data_producer_map_.end()) {
            throw std::runtime_error(
                "Error during TaskGraph construction: Attempted to assign multiple tasks to one data output");
        }

        data_producer_map_[output_handle->ID] = task->ID;
    }

    for (auto &input_handle : task->inputs) {
        // Marks the data as unfulfilled if no task has currently promised to produce it
        if (data_producer_map_.find(input_handle->ID) == data_producer_map_.end()) {
            unfulfilled_data_[input_handle->ID].push_back(task->ID);
            continue;
        }

        int data_producer_id = data_producer_map_[input_handle->ID];
        dependencies_[task->ID].push_back(data_producer_id);
        dependents_[data_producer_id].push_back(task->ID);

        // If the data was previously unfilled, mark it as filled and update the tasks that required it
        if (unfulfilled_data_.find(input_handle->ID) != unfulfilled_data_.end()) {
            for (int unfulfilled_task_id : unfulfilled_data_[input_handle->ID]) {
                dependencies_[unfulfilled_task_id].push_back(task->ID);
                dependents_[task->ID].push_back(unfulfilled_task_id);
            }

            unfulfilled_data_.erase(input_handle->ID);
        }
    }
}

std::vector<int> TaskGraph::find_ready() const {
    std::vector<int> ready_nodes;
    for (auto task_dependencies = dependencies_.begin(); task_dependencies != dependencies_.end();
         ++task_dependencies) {
        if (task_dependencies->second.empty()) {
            ready_nodes.push_back(task_dependencies->first);
        }
    }

    return ready_nodes;
}

std::vector<int> TaskGraph::find_non_ready() const {
    std::vector<int> not_ready_nodes;
    for (auto task_dependencies = dependencies_.begin(); task_dependencies != dependencies_.end();
         ++task_dependencies) {
        if (!task_dependencies->second.empty()) {
            not_ready_nodes.push_back(task_dependencies->first);
        }
    }

    return not_ready_nodes;
}

void TaskGraph::validate_graph() {
    bool no_unfulfilled_data = unfulfilled_data_.empty();
    if (unfulfilled_data_.empty()) {
        // TODO: Refactor to pass back more information about what data was unfulfilled
        throw std::runtime_error("Failed to validate task graph: Data Unfulfillment error");
    }
    std::vector<int> root_nodes = find_ready();

    // Check for cycles utilizing topological sort
    std::deque<int> task_queue(root_nodes.begin(), root_nodes.end());
    std::vector<int> ordering;
    std::unordered_map<int, int> indegree_map;
    for (auto dependent_iter = dependents_.begin(); dependent_iter != dependents_.end(); ++dependent_iter) {
        indegree_map[dependent_iter->first] = dependent_iter->second.size();
    }

    while (!task_queue.empty()) {
        int cur_task = task_queue.front();
        task_queue.pop_front();
        ordering.push_back(cur_task);

        for (int dependent : dependents_[cur_task]) {
            indegree_map[dependent] -= 1;
        }

        for (auto indegree_iter = indegree_map.begin(); indegree_iter != indegree_map.end(); ++indegree_iter) {
            if (indegree_iter->second == 0) {
                task_queue.push_back(indegree_iter->first);
            }
        }
    }

    if (task_queue.size() != all_tasks_.size()) {
        std::runtime_error("Failed to validate task graph: Cyclic task dependency detected");
    }
}

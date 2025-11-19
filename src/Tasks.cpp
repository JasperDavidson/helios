#include "Tasks.h"
#include "Scheduler.h"
#include <deque>
#include <iostream>
#include <mach/task_info.h>
#include <memory>
#include <stdexcept>
#include <unordered_map>

void BaseCPUTask::accept(Scheduler &scheduler) { scheduler.visit(*this); }

void GPUTask::accept(Scheduler &scheduler) { scheduler.visit(*this); }

void TaskGraph::add_task(std::shared_ptr<ITask> task) {
    task->id = task_id_inc++;
    all_tasks_[task->id] = task;

    // Prevent assigning multiple tasks to one output
    if (data_producer_map_.find(task->output_id) != data_producer_map_.end()) {
        throw std::runtime_error(
            "Error during TaskGraph construction: Attempted to assign multiple tasks to one data output");
    }

    // Mapping the output of the task to the task itself
    data_producer_map_[task->output_id] = task->id;

    for (int input_id : task->input_ids) {
        if (data_producer_map_.find(input_id) == data_producer_map_.end()) {
            unfulfilled_data_[input_id].push_back(task->id);
            continue;
        }

        int data_producer_id = data_producer_map_[input_id];
        if (data_producer_id != ROOT_NODE_ID) {
            dependencies_[task->id].push_back(data_producer_id);
            dependents_[data_producer_id].push_back(task->id);
        }
    }

    // If the data was previously unfilled, mark it as filled and update the tasks that required it
    if (unfulfilled_data_.find(task->output_id) != unfulfilled_data_.end()) {
        for (int unfulfilled_task_id : unfulfilled_data_[task->output_id]) {
            dependencies_[unfulfilled_task_id].push_back(task->id);
            dependents_[task->id].push_back(unfulfilled_task_id);
        }

        unfulfilled_data_.erase(task->output_id);
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

std::vector<int> TaskGraph::get_task_ids() const {
    std::vector<int> task_ids;
    for (auto task_iter = all_tasks_.begin(); task_iter != all_tasks_.end(); ++task_iter) {
        task_ids.push_back(task_iter->first);
    }

    return task_ids;
}

void TaskGraph::validate_graph() {
    if (!unfulfilled_data_.empty()) {
        // TODO: Refactor to pass back more information about what data was unfulfilled
        throw std::runtime_error("Failed to validate task graph: Data Unfulfillment error");
    }
    std::vector<int> root_nodes = dependents_[ROOT_NODE_ID];

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

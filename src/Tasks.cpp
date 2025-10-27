#include "Tasks.h"
#include "Scheduler.h"
#include <stdexcept>

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

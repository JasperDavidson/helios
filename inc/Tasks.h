#ifndef TASK_H
#define TASK_H

#include "DataHandle.h"
#include "IGPUExecutor.h"
#include "Runtime.h"
#include <functional>
#include <future>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

class ITask {
  public:
    int ID;
    std::string task_name;
    std::vector<std::shared_ptr<BaseDataHandle>> inputs;
    std::vector<std::shared_ptr<BaseDataHandle>> outputs;

    ITask(int ID, const std::string &task_name, std::vector<std::shared_ptr<BaseDataHandle>> inputs,
          std::vector<std::shared_ptr<BaseDataHandle>> outputs)
        : ID(ID), task_name(task_name), inputs(inputs), outputs(outputs) {};
    ITask() = default;
};

template <typename F, class... Types> class CPUTask : public ITask {
  public:
    std::function<void()> task_lambda;

    CPUTask(int ID, std::string task_name, std::vector<std::shared_ptr<DataHandle<Types...>>> inputs,
            std::vector<std::shared_ptr<DataHandle<Types...>>> outputs, F &&task, const Runtime &runtime)
        : ITask(ID, task_name, inputs, outputs) {
        // Two key things
        // 1. Work lambda to fetch data and run method
        // 2. Wrap that lambda in a packaged task and call the future when CPUTask is executed
        auto work_lambda = [&task, &inputs, &runtime]() {
            auto data = runtime.get_data<Types...>(inputs);
            std::apply(std::forward<F>(task), std::make_tuple(data));
        };

        using TaskReturnType = std::invoke_result<F, Types...>;
        std::shared_ptr<std::packaged_task<TaskReturnType>> task_package =
            std::make_shared<std::packaged_task<TaskReturnType>>(work_lambda);

        task_lambda = [task_package]() { task_package->get_future(); };
    };
};

class GPUTask : public ITask {
  public:
    KernelDispatch task_kernel;

    GPUTask(int ID, std::string task_name, std::vector<BaseDataHandle> inputs, std::vector<BaseDataHandle> outputs);
};

#endif

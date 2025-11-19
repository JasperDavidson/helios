#include "DataManager.h"
#include "Runtime.h"
#include "Tasks.h"
#include <iostream>
#include <vector>

int dot_product(const std::vector<int> &vec1, const std::vector<int> &vec2) {
    int result = 0;
    for (int i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    std::cout << "Dot product result: " << result << std::endl;
    return result;
}

void vec_sum(const std::vector<float> &vec1, const std::vector<float> &vec2, std::vector<float> &output) {
    for (int i = 0; i < vec1.size(); ++i) {
        output.push_back(vec1[i] + vec2[i]);
    }
}

int main() {
    DataManager data_manager;

    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {-1, -2, -3, -4, -5};
    int dot_result = 1;

    std::vector<DataUsage> data_usages = {DataUsage::ReadOnly, DataUsage::ReadOnly, DataUsage::ReadWrite};

    auto vec1_handle = data_manager.create_data_handle(vec1, DataUsage::ReadOnly);
    auto vec2_handle = data_manager.create_data_handle(vec2, DataUsage::ReadOnly);
    auto dot_product_out = data_manager.create_ref_handle(&dot_result);

    std::vector<int> dot_product_in = {vec1_handle.id, vec2_handle.id};
    auto dot_product_task = TypedCPUTask("dot_product", dot_product_in, dot_product_out.id, data_manager, data_usages,
                                         dot_product, vec1_handle, vec2_handle);
    TaskGraph task_graph(dot_product_in);
    task_graph.add_task(std::make_shared<decltype(dot_product_task)>(dot_product_task));

    Runtime helios_runtime(data_manager, 4);
    GPUDevice gpu_device(GPUBackend::Cuda);
    helios_runtime.commit_graph(task_graph, gpu_device);

    std::cout << "Dot product result: " << dot_result << std::endl;

    return 0;
}

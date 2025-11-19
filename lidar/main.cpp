#include "DataManager.h"
#include "Runtime.h"
#include "Tasks.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <vector>

float dot_product(const std::vector<float> &vec1, const std::vector<float> &vec2) {
    float result = 0;
    for (int i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

void vec_sum(const std::vector<float> &vec1, const std::vector<float> &vec2, std::vector<float> &output) {
    for (int i = 0; i < vec1.size(); ++i) {
        output.push_back(vec1[i] + vec2[i]);
    }
}

// Functions for benchmarking
std::vector<float> generate_random_vec(size_t size) {
    std::vector<float> v(size);

    for (int i = 0; i < size; ++i) {
        v[i] = (i * 0.1);
    }

    return v;
}

void benchmark_dot_product(int n_tasks, int num_threads, const std::vector<float> &vec1, const std::vector<float> &vec2,
                           bool print) {
    DataManager data_manager;
    auto vec1_handle = data_manager.create_data_handle(vec1, DataUsage::ReadOnly);
    auto vec2_handle = data_manager.create_data_handle(vec2, DataUsage::ReadOnly);

    std::vector<int> dp_id_in = {vec1_handle.id, vec2_handle.id};
    std::vector<DataUsage> data_usages = {DataUsage::ReadOnly, DataUsage::ReadOnly, DataUsage::ReadWrite};

    TaskGraph task_graph(dp_id_in);
    for (int i = 0; i < n_tasks; ++i) {
        float dp_out = 0;
        auto dp_out_handle = data_manager.create_data_handle(&dp_out);

        std::string name = "dp" + std::to_string(i);
        auto dp_task = TypedCPUTask(name, dp_id_in, dp_out_handle.id, data_manager, data_usages, dot_product,
                                    vec1_handle, vec2_handle);

        task_graph.add_task(std::make_shared<decltype(dp_task)>(dp_task));
    }

    Runtime helios_runtime(data_manager, num_threads);
    GPUDevice device(GPUBackend::Cuda);

    auto start = std::chrono::steady_clock::now();

    helios_runtime.commit_graph(task_graph, device);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (print) {
        std::cout << num_threads << " threads execution time: " << duration << std::endl;
    }
}

int main() {
    // BENCHMARKING CODE
    int n_tasks = 10000;
    size_t vector_size = 1000000;
    int num_threads = 1;

    std::cout << "Generating data...";
    std::vector<float> vec1 = generate_random_vec(vector_size);
    std::vector<float> vec2 = generate_random_vec(vector_size);
    std::cout << "Data generated!\n";

    std::cout << "Running dot product benchmark with " << n_tasks << " tasks on " << num_threads << " threads\n";

    benchmark_dot_product(n_tasks, num_threads, vec1, vec2, true);

    return 0;
}

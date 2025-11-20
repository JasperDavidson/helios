#include "DataManager.h"
#include "Runtime.h"
#include "Tasks.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <vector>

const size_t MACBOOK_PROCESS_THREADS = 10;

float dot_product(const std::vector<float> &vec1, const std::vector<float> &vec2) {
    float result = 0;
    for (int i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

std::vector<float> vec_sum(const std::vector<float> vec1, const std::vector<float> vec2) {
    std::vector<float> result(vec1.size());
    for (int i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

// Functions for benchmarking
std::vector<float> generate_random_vec(size_t size) {
    std::vector<float> v(size);

    for (int i = 0; i < size; ++i) {
        v[i] = (i * 0.1);
    }

    return v;
}

template <typename F, class... Types>
void benchmark(DataManager &data_manager, int num_tasks, auto hash_lambda, F &&task, Types &&...args) {
    std::vector<int> input_ids(sizeof...(args));
    (input_ids.push_back(args.id), ...);

    auto benchmarked_func = std::decay_t<F>(task);
    auto args_tuple = std::make_tuple(std::forward<Types>(args)...);
    auto inputs = std::apply(
        [&](auto &&...handles) { return std::forward_as_tuple(data_manager.get_data(handles)...); }, args_tuple);

    std::cout << "Benchmarking without Helios" << std::endl;
    auto start = std::chrono::steady_clock::now();

    using ResultType = decltype(std::apply(benchmarked_func, inputs));
    ResultType result;
    for (int i = 0; i < num_tasks; ++i) {
        result = hash_lambda(std::apply(benchmarked_func, inputs));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration << "\n\n";

    for (int num_threads = 2; num_threads <= MACBOOK_PROCESS_THREADS; ++num_threads) {
        Runtime helios_runtime(data_manager, num_threads);
        GPUDevice device(GPUBackend::Cuda);

        TaskGraph task_graph;

        std::vector<ResultType> results(num_tasks);

        std::cout << "(Helios) Benchmarking with " << num_threads << " threads" << std::endl;
        for (int n_task = 0; n_task < num_tasks; ++n_task) {
            auto result_handle = data_manager.create_ref_handle(&(results[n_task]));
            std::string name = "benchmark" + std::to_string(n_task);
            auto benchmark_task = TypedCPUTask(name, input_ids, result_handle.id, data_manager, task, args...);

            task_graph.add_task(std::make_shared<decltype(benchmark_task)>(benchmark_task), true);
        }

        auto start = std::chrono::steady_clock::now();

        helios_runtime.commit_graph(task_graph, device);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Execution time: " << duration << "\n\n";

        for (ResultType helios_result : results) {
            if (result != helios_result) {
                throw std::runtime_error("Helios result differed from expected!");
            }
        }
    }
}

void dp_benchmark() {
    int num_tasks = 1000;
    size_t vector_size = 1000000;

    std::cout << "\n\nBENCHMARK: Dot Product\n\n";

    std::cout << "Generating data...";
    std::vector<float> vec1 = generate_random_vec(vector_size);
    std::vector<float> vec2 = generate_random_vec(vector_size);
    std::cout << "Data generated!\n\n";

    DataManager data_manager;
    auto vec1_handle = data_manager.create_data_handle(vec1, DataUsage::ReadOnly);
    auto vec2_handle = data_manager.create_data_handle(vec2, DataUsage::ReadOnly);

    auto hash_lambda = [](float result) { return result; };

    benchmark(data_manager, num_tasks, hash_lambda, dot_product, vec1_handle, vec2_handle);
}

void vec_sum_benchmark() {
    int num_tasks = 1000;
    size_t vector_size = 1000000;

    std::cout << "\n\nBENCHMARK: Vector Sum\n\n";

    std::cout << "Generating data...";
    std::vector<float> vec1 = generate_random_vec(vector_size);
    std::vector<float> vec2 = generate_random_vec(vector_size);
    std::cout << "Data generated!\n\n";

    DataManager data_manager;
    auto vec1_handle = data_manager.create_data_handle(vec1, DataUsage::ReadOnly);
    auto vec2_handle = data_manager.create_data_handle(vec2, DataUsage::ReadOnly);

    auto hash_lambda = [](std::vector<float> vec_sum) { return vec_sum; };

    benchmark(data_manager, num_tasks, hash_lambda, vec_sum, vec1_handle, vec2_handle);
}

int main() {
    dp_benchmark();
    vec_sum_benchmark();

    return 0;
}

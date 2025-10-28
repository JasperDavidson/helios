#include "Runtime.h"
#include "Tasks.h"
#include <future>

std::future<void> Runtime::commit_graph(TaskGraph &task_graph) { task_graph.validate_graph(); };

#include "Runtime.h"
#include "Tasks.h"
#include <future>

// TODO: Figure out return type here
std::future<void> Runtime::commit_graph(TaskGraph &task_graph) { task_graph.validate_graph(); };

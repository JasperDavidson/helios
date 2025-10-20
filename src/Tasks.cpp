#include "Tasks.h"
#include "Scheduler.h"

template <typename F, class... Types> void CPUTask<F, Types...>::accept(const Scheduler &scheduler) {
    scheduler.visit(*this);
}

void GPUTask::accept(const Scheduler &scheduler) { scheduler.visit(*this); }

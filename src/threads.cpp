#include <iostream>
#include <mutex>
#include <atomic>
#include <thread>

// Key idea with mutex is that you can lock a certain system such that two threads can't access it/modify it at once. This helps prevent data races
// Manually locking and unlocking is unsafe --> can lead to deadlock

// Updating a value is actually read -> update value -> write
// If two threads try to do this for the same variable they might interleave and cause issues
// Atomic fixes this by *actually* bundling these 3 tasks into one instruction, "atomizing" it

void print_thread_id_func(std::mutex& m, int id) {
  // Lock guard offers an RAII solution (lock creating on construction and destroyed on destruction)
  std::lock_guard<std::mutex> lg(m);
  std::cout << "Printing from thread id: " << id << '\n';
}

void increment_atomic(std::atomic<int>& counter) {
  for (int i = 0; i < 100; ++i) {
    counter += 1;
  }
}

int main() { 
  return 0;
}

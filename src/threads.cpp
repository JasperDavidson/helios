#include <iostream>
#include <thread>

void print_thread_id(int id) {
  std::cout << "Reading from thread: " << id << '\n';
}

int main() {
  std::jthread t1(print_thread_id, 0);

  return 0;
}

#import <Metal/Metal.h>

#include <iostream>

int main() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::cout << "Hello from CPU" << '\n';

  return 0;
}

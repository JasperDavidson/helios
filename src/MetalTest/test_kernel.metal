#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                       device float *C [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}

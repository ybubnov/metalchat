// vi: set filetype=cpp :

#include <metal_stdlib>


using namespace metal;


kernel void
mul(
    device const float* input,
    device const float* other,
    device float* output,
    uint index [[thread_position_in_grid]]
)
{
    output[index] = input[index] * other[index];
}

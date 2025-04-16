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


kernel void
embedding_bf16(
    device const int32_t* input,
    device const bfloat* weight,
    device bfloat* output,
    uint index [[thread_position_in_grid]]
)
{
    output[index] = weight[input[index]];
}

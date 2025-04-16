// vi: set filetype=cpp :

#include <metal_common>


using namespace metal;


kernel void
mul_bf16(
    constant uint& N [[buffer(0)]],
    device const bfloat* input1 [[buffer(1)]],
    device const bfloat* input2 [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < N) {
        output[gid] = input1[gid] * input2[gid];
    }
}

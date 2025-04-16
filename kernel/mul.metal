// vi: set filetype=cpp :

#include <metal_common>


using namespace metal;


kernel void
mul2d_bf16(
    constant uint& M [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    device const bfloat* input1 [[buffer(2)]],
    device const bfloat* input2 [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint i = gid.x;
    const uint j = gid.y;

    if (i < M && j < N) {
        output[i * N + j] = input1[i * N + j] * input2[i * N + j];
    }
}

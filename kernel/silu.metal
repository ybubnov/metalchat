// vi: set filetype=cpp :

#include <metal_stdlib>


using namespace metal;


kernel void
silu_bf16(
    constant uint& M [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    device const bfloat* input [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint i = gid.x;
    const uint j = gid.y;

    if (i < M && j < N) {
        output[i * N + j] = input[i * N + j] / (bfloat(1.0) + bfloat(exp(-input[i * N + j])));
    }
}

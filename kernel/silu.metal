// vi: set filetype=cpp :

#include <metal_stdlib>


using namespace metal;


kernel void
silu_bf16(
    constant uint& N [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < N) {
        output[gid] = input[gid] / (bfloat(1.0) + bfloat(exp(-input[gid])));
    }
}

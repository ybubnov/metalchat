// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


using namespace metal;


kernel void
sgemm_bf16(
    constant uint& M [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    device const bfloat* input [[buffer(3)]],
    device const bfloat* weight [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint i = gid.x;
    const uint j = gid.y;

    if (i < M && j < N) {
        bfloat partial_sum = bfloat(0.0);
        for (uint k = 0; k < K; ++k) {
            partial_sum += input[i * K + k] * weight[k * N + j];
        }

        output[i * N + j] = partial_sum + output[i * N + j];
    }
}

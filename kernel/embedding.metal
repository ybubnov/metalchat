// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


kernel void
embedding_f16(
    device const int32_t* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]
)
{
    output[index.x * stride + index.y] = weight[input[index.x] * stride + index.y];
}

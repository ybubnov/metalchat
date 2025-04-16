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
    device const int32_t* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const int64_t* stride [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]
)
{
    output[index.x * stride[0] + index.y] = weight[input[index.x] * stride[0] + index.y];
}


//kernel void
//rmsnorm_bf16(
//    device const bfloat* weight [[buffer(0)]],
//    device const bfloat& eps [[buffer(1)]],
//    device bfloat* output [[buffer(3)]]
//    uint index [[thread_position_in_grid]]
//)
//{
//    output[index] = weight[index] * rsqrt(1.0 + eps);
//}

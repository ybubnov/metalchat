// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


#define __embedding_parameters(T)              \
    device const int32_t* input [[buffer(0)]], \
    device const T* weight [[buffer(1)]],      \
    constant uint& stride [[buffer(2)]],       \
    device T* output [[buffer(3)]],            \
    uint2 index [[thread_position_in_grid]]


template <typename T>
kernel void
embedding(__embedding_parameters(T))
{
    output[index.x * stride + index.y] = weight[input[index.x] * stride + index.y];
}


template [[host_name("embedding_bf16")]]
kernel void embedding<bfloat>(__embedding_parameters(bfloat));


template [[host_name("embedding_float")]]
kernel void embedding<float>(__embedding_parameters(float));

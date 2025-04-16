// vi: set filetype=cpp :

#include <metal_stdlib>


using namespace metal;


#define __silu_parameters(T)                   \
    constant uint& dim_size [[buffer(0)]],     \
    device const T* input [[buffer(1)]],       \
    device T* output [[buffer(2)]],            \
    uint gid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
silu(__silu_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    device const T* in = input + gid * dim_size;
    device T* out = output + gid * dim_size;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        T x = in[i + j];
        out[i + j] = x / (T(1.0) + T(exp(-x)));
    }
}


template [[host_name("silu_bf16")]]
kernel void silu<bfloat>(__silu_parameters(bfloat));


template [[host_name("silu_float")]]
kernel void silu<float>(__silu_parameters(float));

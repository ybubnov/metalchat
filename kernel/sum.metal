// vi: set filetype=cpp :

#include <metal_common>


using namespace metal;


#define __sum_parameters(T)                    \
    constant uint& dim_size [[buffer(0)]],     \
    device const T* input1 [[buffer(1)]],      \
    device const T* input2 [[buffer(2)]],      \
    device T* output [[buffer(3)]],            \
    uint gid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
sum(__sum_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    device const T* in1 = input1 + gid * dim_size;
    device const T* in2 = input2 + gid * dim_size;
    device T* out = output + gid * dim_size;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        out[i + j] = in1[i + j] + in2[i + j];
    }
}


template [[host_name("sum_bf16")]]
kernel void sum<bfloat>(__sum_parameters(bfloat));


template [[host_name("sum_float")]]
kernel void sum<float>(__sum_parameters(float));

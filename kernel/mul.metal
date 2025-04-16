// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


using namespace metal;


#define __hadamard_parameters(T)               \
    constant uint& dim_size [[buffer(0)]],     \
    device const T* input1 [[buffer(1)]],      \
    device const T* input2 [[buffer(2)]],      \
    device T* output [[buffer(3)]],            \
    uint gid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
hadamard(__hadamard_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    device const T* in1 = input1 + gid * dim_size;
    device const T* in2 = input2 + gid * dim_size;
    device T* out = output + gid * dim_size;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        out[i + j] = in1[i + j] * in2[i + j];
    }
}


template [[host_name("hadamard_bf16")]]
kernel void hadamard<bfloat>(__hadamard_parameters(bfloat));


template [[host_name("hadamard_float")]]
kernel void hadamard<float>(__hadamard_parameters(float));


#define __scalar_mul_parameters(T)                          \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    constant const T& multiplier             [[buffer(4)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
scalar_mul(__scalar_mul_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, k) * multiplier;
    }
}


template [[host_name("scalar_mul_bf16")]]
kernel void scalar_mul<bfloat>(__scalar_mul_parameters(bfloat));


template [[host_name("scalar_mul_float")]]
kernel void scalar_mul<float>(__scalar_mul_parameters(float));

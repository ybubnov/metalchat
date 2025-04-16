// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


using namespace metal;


#define __hadamard_parameters(T)                             \
    constant tensor_layout<2>& output_layout  [[buffer(0)]], \
    device T* output                          [[buffer(1)]], \
    constant tensor_layout<2>& input1_layout  [[buffer(2)]], \
    device const T* input1                    [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout  [[buffer(4)]], \
    device const T* input2                    [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],               \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
hadamard(__hadamard_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) * in2.at(i, k);
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

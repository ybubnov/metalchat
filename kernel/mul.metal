// vi: set filetype=cpp :

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __hadamard_parameters {
    constant layout2& output_layout;
    device T* output_data;
    constant layout2& input1_layout;
    const device T* input1_data;
    constant layout2& input2_layout;
    const device T* input2_data;
};


template <typename T, uint BlockSize>
kernel void
hadamard(
    __hadamard_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    tensor2<T> output(params.output_layout, params.output_data);
    tensor2<const T> input1(params.input1_layout, params.input1_data);
    tensor2<const T> input2(params.input2_layout, params.input2_data);

    const uint dim_size = input1.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

#pragma unroll
    for (uint k = 0; k < end && k < dim_size; k++) {
        output.at(i, k) = input1.at(i, k) * input2.at(i, k);
    }
}


__lib_metalchat_kernel(hadamard, bfloat, 8);
__lib_metalchat_kernel(hadamard, bfloat, 16);
__lib_metalchat_kernel(hadamard, bfloat, 32);

__lib_metalchat_kernel(hadamard, float, 8);
__lib_metalchat_kernel(hadamard, float, 16);
__lib_metalchat_kernel(hadamard, float, 32);


template <typename T> struct __scalar_mul_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& input_layout;
    device const T* input;
    constant const T& multiplier;
};


template <typename T, uint BlockSize>
kernel void
scalar_mul(
    __scalar_mul_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    tensor<const T, 2> in{params.input, params.input_layout};
    tensor<T, 2> out{params.output, params.output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

#pragma unroll
    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, k) * params.multiplier;
    }
}


__lib_metalchat_kernel(scalar_mul, bfloat, 8);
__lib_metalchat_kernel(scalar_mul, bfloat, 16);
__lib_metalchat_kernel(scalar_mul, bfloat, 32);
__lib_metalchat_kernel(scalar_mul, bfloat, 128);

__lib_metalchat_kernel(scalar_mul, float, 8);
__lib_metalchat_kernel(scalar_mul, float, 16);
__lib_metalchat_kernel(scalar_mul, float, 32);

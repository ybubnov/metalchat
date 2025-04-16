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
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor2<T> output(params.output_layout, params.output_data);
    tensor2<const T> input1(params.input1_layout, params.input1_data);
    tensor2<const T> input2(params.input2_layout, params.input2_data);

    const uint dim_size = input1.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        output.at(i, k) = input1.at(i, k) * input2.at(i, k);
    }
}


__lib_metalchat_kernel2(hadamard, bfloat, 8);
__lib_metalchat_kernel2(hadamard, bfloat, 16);
__lib_metalchat_kernel2(hadamard, bfloat, 32);

__lib_metalchat_kernel2(hadamard, float, 8);
__lib_metalchat_kernel2(hadamard, float, 16);
__lib_metalchat_kernel2(hadamard, float, 32);


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
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor<const T, 2> in{params.input, params.input_layout};
    tensor<T, 2> out{params.output, params.output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        out.at(i, k) = in.at(i, k) * params.multiplier;
    }
}


__lib_metalchat_kernel2(scalar_mul, bfloat, 1);
__lib_metalchat_kernel2(scalar_mul, bfloat, 8);
__lib_metalchat_kernel2(scalar_mul, bfloat, 16);
__lib_metalchat_kernel2(scalar_mul, bfloat, 32);
__lib_metalchat_kernel2(scalar_mul, bfloat, 128);

__lib_metalchat_kernel2(scalar_mul, float, 8);
__lib_metalchat_kernel2(scalar_mul, float, 16);
__lib_metalchat_kernel2(scalar_mul, float, 32);

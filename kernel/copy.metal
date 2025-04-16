// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __copy_parameters(T)                                \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output [[buffer(1)]],                         \
    constant tensor_layout<2>& input_layout [[buffer(2)]],  \
    device const T* input [[buffer(3)]],                    \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_position_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
copy(__copy_parameters(T))
{
    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, k);
    }
}


template [[host_name("copy_1_bf16")]]
kernel void copy<bfloat, 1>(__copy_parameters(bfloat));

template [[host_name("copy_2_bf16")]]
kernel void copy<bfloat, 2>(__copy_parameters(bfloat));

template [[host_name("copy_4_bf16")]]
kernel void copy<bfloat, 4>(__copy_parameters(bfloat));

template [[host_name("copy_8_bf16")]]
kernel void copy<bfloat, 8>(__copy_parameters(bfloat));

template [[host_name("copy_16_bf16")]]
kernel void copy<bfloat, 16>(__copy_parameters(bfloat));

template [[host_name("copy_32_bf16")]]
kernel void copy<bfloat, 32>(__copy_parameters(bfloat));


template [[host_name("copy_1_float")]]
kernel void copy<float, 1>(__copy_parameters(float));

template [[host_name("copy_2_float")]]
kernel void copy<float, 2>(__copy_parameters(float));

template [[host_name("copy_4_float")]]
kernel void copy<float, 4>(__copy_parameters(float));

template [[host_name("copy_8_float")]]
kernel void copy<float, 8>(__copy_parameters(float));

template [[host_name("copy_16_float")]]
kernel void copy<float, 16>(__copy_parameters(float));

template [[host_name("copy_32_float")]]
kernel void copy<float, 32>(__copy_parameters(float));

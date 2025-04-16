// vi: set filetype=cpp :

#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __silu_parameters(T)                                \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_index_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
silu(__silu_parameters(T))
{
    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        T x = in.at(i, k);
        out.at(i, k) = x / (T(1.0) + T(exp(-x)));
    }
}


template [[host_name("silu_1_bf16")]]
kernel void silu<bfloat, 1>(__silu_parameters(bfloat));

template [[host_name("silu_2_bf16")]]
kernel void silu<bfloat, 2>(__silu_parameters(bfloat));

template [[host_name("silu_4_bf16")]]
kernel void silu<bfloat, 4>(__silu_parameters(bfloat));

template [[host_name("silu_8_bf16")]]
kernel void silu<bfloat, 8>(__silu_parameters(bfloat));

template [[host_name("silu_16_bf16")]]
kernel void silu<bfloat, 16>(__silu_parameters(bfloat));

template [[host_name("silu_32_bf16")]]
kernel void silu<bfloat, 32>(__silu_parameters(bfloat));


template [[host_name("silu_1_float")]]
kernel void silu<float, 1>(__silu_parameters(float));

template [[host_name("silu_2_float")]]
kernel void silu<float, 2>(__silu_parameters(float));

template [[host_name("silu_4_float")]]
kernel void silu<float, 4>(__silu_parameters(float));

template [[host_name("silu_8_float")]]
kernel void silu<float, 8>(__silu_parameters(float));

template [[host_name("silu_16_float")]]
kernel void silu<float, 16>(__silu_parameters(float));

template [[host_name("silu_32_float")]]
kernel void silu<float, 32>(__silu_parameters(float));

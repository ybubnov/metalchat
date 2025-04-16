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


template <typename T>
kernel void
silu(__silu_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint k = begin; k < end && k < dim_size; k++) {
        T x = in.at(i, k);
        out.at(i, k) = x / (T(1.0) + T(exp(-x)));
    }
}


template [[host_name("silu_bf16")]]
kernel void silu<bfloat>(__silu_parameters(bfloat));


template [[host_name("silu_float")]]
kernel void silu<float>(__silu_parameters(float));

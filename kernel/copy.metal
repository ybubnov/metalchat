// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __copy_parameters(T)                                \
    constant tensor_layout<2>& input_layout [[buffer(0)]],  \
    constant tensor_layout<2>& output_layout [[buffer(1)]], \
    device const T* input [[buffer(2)]],                    \
    device T* output [[buffer(3)]],                         \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_position_in_threadgroup]]


template <typename T>
kernel void
copy(__copy_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, k);
    }
}


template [[host_name("copy_bf16")]]
kernel void copy<bfloat>(__copy_parameters(bfloat));


template [[host_name("copy_float")]]
kernel void copy<float>(__copy_parameters(float));

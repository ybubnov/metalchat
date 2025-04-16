// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __copy_parameters {
    constant tensor_layout<2>& output_layout [[buffer(0)]];
    device T* output [[buffer(1)]];
    constant tensor_layout<2>& input_layout [[buffer(2)]];
    device const T* input [[buffer(3)]];
};


template <typename T, uint BlockSize>
kernel void
copy(
    __copy_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    tensor<const T, 2> in{params.input, params.input_layout};
    tensor<T, 2> out{params.output, params.output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, k);
    }
}


__lib_metalchat_kernel(copy, bfloat, 8);
__lib_metalchat_kernel(copy, bfloat, 16);
__lib_metalchat_kernel(copy, bfloat, 32);

__lib_metalchat_kernel(copy, float, 8);
__lib_metalchat_kernel(copy, float, 16);
__lib_metalchat_kernel(copy, float, 32);

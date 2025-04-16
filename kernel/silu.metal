// vi: set filetype=cpp :

#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __silu_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T, uint BlockSize>
kernel void
silu(
    __silu_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        T x = params.input.at(i, k);
        params.output.at(i, k) = x / (T(1.0) + T(exp(-x)));
    }
}


__lib_metalchat_kernel(silu, bfloat, 8);
__lib_metalchat_kernel(silu, bfloat, 16);
__lib_metalchat_kernel(silu, bfloat, 32);

__lib_metalchat_kernel(silu, float, 8);
__lib_metalchat_kernel(silu, float, 16);
__lib_metalchat_kernel(silu, float, 32);

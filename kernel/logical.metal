// vi: set filetype=cpp :

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __gt_parameters {
    tensor2<bool> output;
    tensor2<const T> input;
    constant T& value;
};


template <typename T, uint BlockSize>
kernel void
gt(__gt_parameters<T> params,
   uint gid [[threadgroup_position_in_grid]],
   uint tid [[thread_index_in_threadgroup]])
{
    const uint dim_size = params.input.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = 0; k < end && k < dim_size; k++) {
        params.output.at(i, k) = (params.input.at(i, k) > params.value);
    }
}


__lib_metalchat_kernel(gt, bfloat, 8);
__lib_metalchat_kernel(gt, bfloat, 16);
__lib_metalchat_kernel(gt, bfloat, 32);
__lib_metalchat_kernel(gt, bfloat, 256);

__lib_metalchat_kernel(gt, float, 8);
__lib_metalchat_kernel(gt, float, 16);
__lib_metalchat_kernel(gt, float, 32);

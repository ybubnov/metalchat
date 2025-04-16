// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __copy_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T, uint BlockSize>
kernel void
copy(
    __copy_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        params.output.at(i, k) = params.input.at(i, k);
    }
}


__lib_metalchat_kernel(copy, bfloat, 8);
__lib_metalchat_kernel(copy, bfloat, 16);
__lib_metalchat_kernel(copy, bfloat, 32);

__lib_metalchat_kernel(copy, float, 8);
__lib_metalchat_kernel(copy, float, 16);
__lib_metalchat_kernel(copy, float, 32);


template <typename T> struct __scatter_parameters {
    tensor2<T> output;
    tensor2<const bool> mask;
    constant T& value;
};


template <typename T, uint BlockSize>
kernel void
scatter(
    __scatter_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    const uint dim_size = params.output.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        if (params.mask.at(i, k)) {
            params.output.at(i, k) = params.value;
        }
    }
}

__lib_metalchat_kernel(scatter, bfloat, 8);
__lib_metalchat_kernel(scatter, bfloat, 16);
__lib_metalchat_kernel(scatter, bfloat, 32);

__lib_metalchat_kernel(scatter, float, 8);
__lib_metalchat_kernel(scatter, float, 16);
__lib_metalchat_kernel(scatter, float, 32);


template <typename T> struct __gather_parameters {
    tensor2<T> output;
    tensor2<const T> input;
    tensor2<const int32_t> index;
};


template <typename T, uint BlockSize>
kernel void
gather(
    __gather_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    const uint dim_size = params.index.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        params.output.at(i, k) = params.input.at(i, params.index.at(i, k));
    }
}

__lib_metalchat_kernel(gather, bfloat, 8);
__lib_metalchat_kernel(gather, bfloat, 16);
__lib_metalchat_kernel(gather, bfloat, 32);

__lib_metalchat_kernel(gather, float, 8);
__lib_metalchat_kernel(gather, float, 16);
__lib_metalchat_kernel(gather, float, 32);

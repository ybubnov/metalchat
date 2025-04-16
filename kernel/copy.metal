// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __copy_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& input_layout;
    device const T* input;
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


template <typename T> struct __inplace_index_set_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& mask_layout;
    device const bool* mask;
    constant T& value;
};


template <typename T, uint BlockSize>
kernel void
inplace_index_set(
    __inplace_index_set_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    tensor<T, 2> out{params.output, params.output_layout};
    tensor<const bool, 2> m{params.mask, params.mask_layout};

    const uint dim_size = out.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        if (m.at(i, k)) {
            out.at(i, k) = params.value;
        }
    }
}

__lib_metalchat_kernel(inplace_index_set, bfloat, 8);
__lib_metalchat_kernel(inplace_index_set, bfloat, 16);
__lib_metalchat_kernel(inplace_index_set, bfloat, 32);

__lib_metalchat_kernel(inplace_index_set, float, 8);
__lib_metalchat_kernel(inplace_index_set, float, 16);
__lib_metalchat_kernel(inplace_index_set, float, 32);

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


template <typename T> struct __scatter_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& mask_layout;
    device const bool* mask;
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

__lib_metalchat_kernel(scatter, bfloat, 8);
__lib_metalchat_kernel(scatter, bfloat, 16);
__lib_metalchat_kernel(scatter, bfloat, 32);

__lib_metalchat_kernel(scatter, float, 8);
__lib_metalchat_kernel(scatter, float, 16);
__lib_metalchat_kernel(scatter, float, 32);


template <typename T> struct __gather_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& input_layout;
    device const T* input;
    constant tensor_layout<2>& index_layout;
    device const int32_t* index;
};


template <typename T, uint BlockSize>
kernel void
gather(
    __gather_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    tensor<T, 2> out{params.output, params.output_layout};
    tensor<const T, 2> in{params.input, params.input_layout};
    tensor<const int32_t, 2> idx{params.index, params.index_layout};

    const uint dim_size = idx.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(i, k) = in.at(i, idx.at(i, k));
    }
}

__lib_metalchat_kernel(gather, bfloat, 8);
__lib_metalchat_kernel(gather, bfloat, 16);
__lib_metalchat_kernel(gather, bfloat, 32);

__lib_metalchat_kernel(gather, float, 8);
__lib_metalchat_kernel(gather, float, 16);
__lib_metalchat_kernel(gather, float, 32);

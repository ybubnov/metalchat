// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __copy_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T>
kernel void
copy(
    __copy_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        params.output.at(i, k) = params.input.at(i, k);
    }
}


__lib_metalchat_kernel2(copy, bfloat);
__lib_metalchat_kernel2(copy, float);


template <typename T> struct __scatter_parameters {
    tensor2<T> output;
    tensor2<const bool> mask;
    constant T& value;
};


template <typename T>
kernel void
scatter(
    __scatter_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.output.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        if (params.mask.at(i, k)) {
            params.output.at(i, k) = params.value;
        }
    }
}

__lib_metalchat_kernel2(scatter, bfloat);
__lib_metalchat_kernel2(scatter, float);


template <typename T> struct __gather_parameters {
    constant layout2& output_layout;
    device T* output_data;
    constant layout2& input_layout;
    device const T* input_data;
    constant layout2& index_layout;
    device const int32_t* index_data;
};


template <typename T>
kernel void
gather(
    __gather_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor2<T> output(params.output_layout, params.output_data);
    tensor2<const T> input(params.input_layout, params.input_data);
    tensor2<const int32_t> index(params.index_layout, params.index_data);

    const uint dim_size = index.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        output.at(i, k) = input.at(i, index.at(i, k));
    }
}


__lib_metalchat_kernel2(gather, float);
__lib_metalchat_kernel2(gather, int32_t);

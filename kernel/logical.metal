// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __gt_parameters {
    tensor2<bool> output;
    tensor2<const T> input;
    constant T& value;
};


template <typename T>
kernel void
gt(__gt_parameters<T> params,
   uint2 gid [[threadgroup_position_in_grid]],
   uint2 tid [[thread_position_in_threadgroup]],
   uint2 threadgroup_size [[threads_per_threadgroup]])
{
    const uint dim_size = params.input.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        params.output.at(i, k) = (params.input.at(i, k) > params.value);
    }
}


__lib_metalchat_kernel2(gt, bfloat);
__lib_metalchat_kernel2(gt, float);


template <typename T> struct __le_parameters {
    tensor2<bool> output;
    tensor2<const T> input;
    constant T& value;
};


template <typename T>
kernel void
le(__le_parameters<T> params,
   uint2 gid [[threadgroup_position_in_grid]],
   uint2 tid [[thread_position_in_threadgroup]],
   uint2 threadgroup_size [[threads_per_threadgroup]])
{
    const uint dim_size = params.input.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        params.output.at(i, k) = (params.input.at(i, k) <= params.value);
    }
}


__lib_metalchat_kernel2(le, bfloat);
__lib_metalchat_kernel2(le, float);

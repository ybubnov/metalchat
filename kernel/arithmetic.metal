// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __add_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input1);
    StructuredBuffer(T, 2, input2);
};


template <typename T>
kernel void
add(__add_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

    const uint row_size = input1.size(0);
    const uint dim_size = input1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input1.at(i, k) + input2.at(i, k);
    }
}


__lib_metalchat_kernel2(add, bfloat);
__lib_metalchat_kernel2(add, float);


template <typename T> struct __add_broadcast_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input1);
    StructuredBuffer(T, 1, input2);
};


template <typename T>
kernel void
add_broadcast(
    __add_broadcast_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

    const uint dim0_size = input1.size(0);
    const uint dim1_size = input1.size(1);

    const uint j = gid.x * threadgroup_size.x + tid.x;
    const uint i = gid.y * threadgroup_size.y + tid.y;

    if (i < dim0_size && j < dim1_size) {
        output.at(i, j) = input1.at(i, j) + input2.at(j % input2.size(0));
    }
}


__lib_metalchat_kernel2(add_broadcast, bfloat);
__lib_metalchat_kernel2(add_broadcast, float);


template <typename T> struct __sub_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input1);
    StructuredBuffer(T, 2, input2);
};


template <typename T>
kernel void
sub(__sub_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

    const uint row_size = input1.size(0);
    const uint dim_size = input1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input1.at(i, k) - input2.at(i, k);
    }
}


__lib_metalchat_kernel2(sub, bfloat);
__lib_metalchat_kernel2(sub, float);


template <typename T> struct __div_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input1);
    StructuredBuffer(T, 2, input2);
};


template <typename T>
kernel void
div(__div_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

    const uint row_size = input1.size(0);
    const uint dim_size = input1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input1.at(i, k) / input2.at(i, k);
    }
}


__lib_metalchat_kernel2(div, bfloat);
__lib_metalchat_kernel2(div, float);

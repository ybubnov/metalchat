// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __copy_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input);
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
    Tensor(params, output);
    Tensor(params, input);

    const uint row_size = input.size(0);
    const uint dim_size = input.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input.at(i, k);
    }
}


__lib_metalchat_kernel2(copy, bfloat);
__lib_metalchat_kernel2(copy, float);
__lib_metalchat_kernel2(copy, int32_t);


template <typename T> struct __scalar_scatter_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(bool, 2, mask);
    constant T& value;
};


template <typename T>
kernel void
scalar_scatter(
    __scalar_scatter_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, mask);

    const uint row_size = output.size(0);
    const uint dim_size = output.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        if (mask.at(i, k)) {
            output.at(i, k) = params.value;
        }
    }
}

__lib_metalchat_kernel2(scalar_scatter, bfloat);
__lib_metalchat_kernel2(scalar_scatter, float);


template <typename T> struct __scatter_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(int32_t, 2, index);
    StructuredBuffer(T, 2, input);
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
    Tensor(params, output);
    Tensor(params, index);
    Tensor(params, input);

    const uint row_size = index.size(0);
    const uint dim_size = index.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, index.at(i, k)) = input.at(i, k);
    }
}


__lib_metalchat_kernel2(scatter, bfloat);
__lib_metalchat_kernel2(scatter, float);


template <typename T> struct __gather_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input);
    StructuredBuffer(int32_t, 2, index);
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
    Tensor(params, output);
    Tensor(params, input);
    Tensor(params, index);

    const uint row_size = index.size(0);
    const uint dim_size = index.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input.at(i, index.at(i, k));
    }
}


__lib_metalchat_kernel2(gather, float);
__lib_metalchat_kernel2(gather, bfloat);
__lib_metalchat_kernel2(gather, int32_t);


template <typename T> struct __where_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(bool, 2, cond);
    StructuredBuffer(T, 2, input);
    StructuredBuffer(T, 2, other);
};


template <typename T>
kernel void
where(
    __where_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, cond);
    Tensor(params, input);
    Tensor(params, other);

    const uint row_size = cond.size(0);
    const uint dim_size = cond.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = cond.at(i, k) ? input.at(i, k) : other.at(i, k);
    }
}


__lib_metalchat_kernel2(where, float);
__lib_metalchat_kernel2(where, bfloat);
__lib_metalchat_kernel2(where, int32_t);

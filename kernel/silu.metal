// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __silu_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T>
kernel void
silu(
    __silu_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        T x = params.input.at(i, k);
        params.output.at(i, k) = x / (T(1.0) + T(metal::exp(-x)));
    }
}


__lib_metalchat_kernel2(silu, bfloat);
__lib_metalchat_kernel2(silu, float);

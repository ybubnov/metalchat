// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __roll_parameters {
    RWStructuredBuffer(T, 1, output);
    StructuredBuffer(T, 1, input);
    constant uint& shift;
    constant uint& size;
    constant uint& stride;
};


template <typename T>
kernel void
roll(
    __roll_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, input);

    const uint k = gid * threadgroup_size + tid;

    const uint dim_size = params.size;
    const uint stride_size = params.size * params.stride;
    const uint size = (k / stride_size) * stride_size;

    uint i = (k / params.stride + params.shift) % dim_size;
    uint j = k % params.stride;
    uint m = size + i * params.stride + j;

    if (k < input.size(0)) {
        output.at(k) = input.at(m);
    }
}


__lib_metalchat_kernel(roll, bfloat);
__lib_metalchat_kernel(roll, float);

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
    const uint row_size = params.input.size(0);
    const uint dim_size = params.input.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        T x = params.input.at(i, k);
        params.output.at(i, k) = x / (T(1.0) + T(metal::exp(-x)));
    }
}


__lib_metalchat_kernel2(silu, bfloat);
__lib_metalchat_kernel2(silu, float);


template <typename T> struct __gelu_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T>
kernel void
gelu(
    __gelu_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    constexpr float beta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
    constexpr float kappa = 0.044715f;

    const uint row_size = params.input.size(0);
    const uint dim_size = params.input.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        const float x = params.input.at(i, k);
        const float x3 = x * x * x;
        const float inner = beta * (x + kappa * x3);

        params.output.at(i, k) = T(0.5f * x * (1 + metal::precise::tanh(inner)));
    }
}


__lib_metalchat_kernel2(gelu, bfloat);
__lib_metalchat_kernel2(gelu, float);

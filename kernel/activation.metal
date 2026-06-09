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
    constexpr float gelu_limit = 4.0f;
    constexpr float gelu_coeff = 0.044714f;

    const uint row_size = params.input.size(0);
    const uint dim_size = params.input.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        const T x = params.input.at(i, k);

        if (x >= gelu_limit) {
            params.output.at(i, k) = x;
        } else if (x <= -gelu_limit) {
            params.output.at(i, k) = T(0);
        } else {
            const float sqrt_2_pi = metal::fast::sqrt(2.0 / M_PI_F);
            const float xh = x + gelu_coeff * metal::fast::pow(x, 3);

            params.output.at(i, k) = T(x * 0.5 * (1.0 + metal::tanh(sqrt_2_pi * xh)));
        }
    }
}


__lib_metalchat_kernel2(gelu, bfloat);
__lib_metalchat_kernel2(gelu, float);

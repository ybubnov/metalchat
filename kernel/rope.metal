// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __rope_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input);
    StructuredBuffer(float, 2, freqs_cos);
    StructuredBuffer(float, 2, freqs_sin);
    constant uint& batch_size;
    constant uint& n_head;
    constant uint& start_pos;
};


template <typename T>
kernel void
rope(
    __rope_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, input);
    Tensor(params, freqs_cos);
    Tensor(params, freqs_sin);

    const uint row_size = input.size(0);
    const uint head_dim = freqs_cos.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    // inputs size: (bs * seq_len * n_head, head_dim)
    const uint pos = i / (params.batch_size * params.n_head);

    if (i < row_size && k < head_dim) {
        float x1 = input.at(i, k);
        float x2 = input.at(i, head_dim + k);

        float fcos = freqs_cos.at(params.start_pos + pos, k);
        float fsin = freqs_sin.at(params.start_pos + pos, k);

        output.at(i, k) = T(fcos * x1 - fsin * x2);
        output.at(i, head_dim + k) = T(fsin * x1 + fcos * x2);
    }
}


__lib_metalchat_kernel2(rope, bfloat);
__lib_metalchat_kernel2(rope, float);


template <typename T> struct __rope_freqs_parameters {
    RWStructuredBuffer(T, 2, freqs_cos);
    RWStructuredBuffer(T, 2, freqs_sin);
    constant uint& dim;
    constant uint& start_pos;
    constant T& theta;
};


template <typename T>
kernel void
rope_freqs(
    __rope_freqs_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, freqs_cos);
    Tensor(params, freqs_sin);

    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint j = gid.x * threadgroup_size.x + tid.x;

    if (i < freqs_cos.size(0) && j < params.dim / 2) {
        float freq = 1.0f / metal::precise::pow(params.theta, 2.0f * j / params.dim);
        float angle = float(params.start_pos + i) * freq;

        freqs_cos.at(i, j) = T(metal::precise::cos(angle));
        freqs_sin.at(i, j) = T(metal::precise::sin(angle));
    }
}


__lib_metalchat_kernel2(rope_freqs, float);

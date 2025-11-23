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
    tensor2<T> output;
    tensor2<const T> input;
    constant layout2& freqs_cos_layout;
    device const float* freqs_cos;
    constant layout2& freqs_sin_layout;
    device const float* freqs_sin;
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
    tensor2<const float> f_cos(params.freqs_cos_layout, params.freqs_cos);
    tensor2<const float> f_sin(params.freqs_sin_layout, params.freqs_sin);

    const uint head_dim = f_cos.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    // numel = bs * seq_len * n_head * head_dim
    const uint pos = i / (params.batch_size * params.n_head);

    if (k < head_dim) {
        float x1 = params.input.at(i, 2 * k);
        float x2 = params.input.at(i, 2 * k + 1);

        float fcos = f_cos.at(params.start_pos + pos, k);
        float fsin = f_sin.at(params.start_pos + pos, k);

        params.output.at(i, 2 * k) = T(fcos * x1 - fsin * x2);
        params.output.at(i, 2 * k + 1) = T(fsin * x1 + fcos * x2);
    }
}


__lib_metalchat_kernel2(rope, bfloat);
__lib_metalchat_kernel2(rope, float);


template <typename T> struct __rope_freqs_parameters {
    constant layout2& freqs_cos_layout;
    device T* freqs_cos;
    constant layout2& freqs_sin_layout;
    device T* freqs_sin;
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
    tensor2<T> f_cos(params.freqs_cos_layout, params.freqs_cos);
    tensor2<T> f_sin(params.freqs_sin_layout, params.freqs_sin);

    const uint i = gid.x;
    const uint j = tid.x + gid.y * threadgroup_size.x;

    if (j < params.dim / 2) {
        T freq = T(1.0) / metal::pow(params.theta, 2.0 * j / params.dim);
        T angle = T(params.start_pos + i) * freq;

        f_cos.at(i, j) = metal::cos(angle);
        f_sin.at(i, j) = metal::sin(angle);
    }
}


__lib_metalchat_kernel2(rope_freqs, float);

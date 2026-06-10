// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __bmm_parameters {
    constant layout3& output_layout;
    device T* output;
    constant layout3& mat1_layout;
    device const T* mat1;
    constant layout3& mat2_layout;
    device const T* mat2;
};


/// Matrix multiplication mat1(b x M x K) @ mat2(b x K x N) -> C(b x M x N)
template <typename T, uint BlockSize>
kernel void
bmm(__bmm_parameters<T> params,
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]])
{
    tensor3<const T> m1(params.mat1_layout, params.mat1);
    tensor3<const T> m2(params.mat2_layout, params.mat2);
    tensor3<T> out(params.output_layout, params.output);

    const uint M = m1.size(1);
    const uint K = m1.size(2);
    const uint N = m2.size(2);

    threadgroup float m1_local[BlockSize][BlockSize];
    threadgroup float m2_local[BlockSize][BlockSize];

    const uint block_row = group_id.x * BlockSize;
    const uint block_col = group_id.y * BlockSize;

    const uint batch = group_id.z;
    const uint thread_row = thread_id.x;
    const uint thread_col = thread_id.y;

    float partial = 0.0f;

    uint r1 = block_row + thread_row;
    uint c2 = block_col + thread_col;

    for (uint k = 0; k < K; k += BlockSize) {
        uint c1 = k + thread_col;
        m1_local[thread_row][thread_col] = (r1 < M && c1 < K) ? m1.at(batch, r1, c1) : 0.0f;

        uint r2 = k + thread_row;
        m2_local[thread_row][thread_col] = (r2 < K && c2 < N) ? m2.at(batch, r2, c2) : 0.0f;

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#pragma unroll(BlockSize)
        for (uint j = 0; j < BlockSize; ++j) {
            partial += m1_local[thread_row][j] * m2_local[j][thread_col];
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    uint row = block_row + thread_row;
    uint col = block_col + thread_col;

    if (row < M && col < N) {
        out.at(batch, row, col) = T(partial);
    }
}


__lib_metalchat_kernel3_tiled(bmm, 8, bfloat);
__lib_metalchat_kernel3_tiled(bmm, 8, float);

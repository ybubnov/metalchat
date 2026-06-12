// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_simdgroup_matrix>
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
    uint3 threadgroup_size [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[simdgroups_per_threadgroup]])
{
    tensor3<const T> m1(params.mat1_layout, params.mat1);
    tensor3<const T> m2(params.mat2_layout, params.mat2);
    tensor3<T> out(params.output_layout, params.output);

    const uint M = m1.size(1);
    const uint K = m1.size(2);
    const uint N = m2.size(2);

    threadgroup T m1_local[BlockSize][BlockSize];
    threadgroup T m2_local[BlockSize][BlockSize];

    const uint block_row = group_id.x * BlockSize;
    const uint block_col = group_id.y * BlockSize;

    const uint batch = group_id.z;
    const uint thread_row_a = thread_id.x;
    const uint thread_row_b = thread_id.x + threadgroup_size.x;
    const uint thread_col = thread_id.y;

    constexpr uint tile_size = 8;
    constexpr uint tiles_per_simdgroup = BlockSize / tile_size;

    const uint simd_row = (simd_gid / tiles_per_simdgroup) * tile_size;
    const uint simd_col = (simd_gid % tiles_per_simdgroup) * tile_size;

    using SimdTensor = metal::simdgroup_matrix<T, tile_size, tile_size>;
    using SimdData = threadgroup T*;

    thread SimdTensor mm_simd(0);
    thread SimdTensor m1_simd;
    thread SimdTensor m2_simd;

    uint row_a = block_row + thread_row_a;
    uint row_b = block_row + thread_row_b;
    uint col = block_col + thread_col;

    for (uint k = 0; k < K; k += BlockSize) {
        uint c1 = k + thread_col;
        m1_local[thread_row_a][thread_col] = (row_a < M && c1 < K) ? m1.at(batch, row_a, c1) : 0;
        m1_local[thread_row_b][thread_col] = (row_b < M && c1 < K) ? m1.at(batch, row_b, c1) : 0;

        uint r2a = k + thread_row_a;
        uint r2b = k + thread_row_b;
        m2_local[thread_row_a][thread_col] = (r2a < K && col < N) ? m2.at(batch, r2a, col) : 0;
        m2_local[thread_row_b][thread_col] = (r2b < K && col < N) ? m2.at(batch, r2b, col) : 0;

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint j = 0; j < BlockSize; j += tile_size) {
            metal::simdgroup_load(m1_simd, SimdData(m1_local), BlockSize, ulong2(j, simd_row));
            metal::simdgroup_load(m2_simd, SimdData(m2_local), BlockSize, ulong2(simd_col, j));
            metal::simdgroup_multiply_accumulate(mm_simd, m1_simd, m2_simd, mm_simd);
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    metal::simdgroup_store(mm_simd, SimdData(m1_local), BlockSize, ulong2(simd_col, simd_row));
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (row_a < M && col < N) {
        out.at(batch, row_a, col) = m1_local[thread_row_a][thread_col];
    }
    if (row_b < M && col < N) {
        out.at(batch, row_b, col) = m1_local[thread_row_b][thread_col];
    }
}


__lib_metalchat_kernel3_tiled(bmm, 8, bfloat);
__lib_metalchat_kernel3_tiled(bmm, 16, bfloat);
__lib_metalchat_kernel3_tiled(bmm, 32, bfloat);
__lib_metalchat_kernel3_tiled(bmm, 8, float);
__lib_metalchat_kernel3_tiled(bmm, 16, float);
__lib_metalchat_kernel3_tiled(bmm, 32, float);

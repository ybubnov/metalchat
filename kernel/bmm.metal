// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __gemm_parameters {
    tensor3<T> output;
    constant layout3& mat1_layout;
    device const T* mat1;
    constant layout3& mat2_layout;
    device const T* mat2;
};


/// Matrix multiplication mat1(b x M x K) @ mat2(b x K x N) -> C(b x M x N)
///
/// Currently Metal framework supports only SIMD-group matrices of size 8x8, here we for
/// simplicity of other computations (SIMD row and column) keep it as a kernel template
/// parameter.
template <typename T, uint BlockSize, uint TileSize = 8>
kernel void
gemm(
    __gemm_parameters<T> params,
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[simdgroups_per_threadgroup]]
)
{
    tensor3<const T> m1(params.mat1_layout, params.mat1);
    tensor3<const T> m2(params.mat2_layout, params.mat2);

    const uint M = m1.size(1);
    const uint K = m1.size(2);
    const uint N = m2.size(2);

    threadgroup T m1_local[BlockSize][BlockSize];
    threadgroup T m2_local[BlockSize][BlockSize];

    const uint block_row = group_id.x * BlockSize;
    const uint block_col = group_id.y * BlockSize;

    const uint batch = group_id.z;
    const uint thread_row = thread_id.x;
    const uint thread_col = thread_id.y;

    constexpr uint tiles_per_simdgroup = BlockSize / TileSize;

    const uint simd_row = (simd_gid / tiles_per_simdgroup) * TileSize;
    const uint simd_col = (simd_gid % tiles_per_simdgroup) * TileSize;

    using SimdTensor = metal::simdgroup_matrix<T, TileSize, TileSize>;
    using SimdData = threadgroup T*;

    SimdTensor mm_simd(0);
    SimdTensor m1_simd;
    SimdTensor m2_simd;

    uint row = block_row + thread_row;
    uint col = block_col + thread_col;

    for (uint k = 0; k < K; k += BlockSize) {
        uint col_k = k + thread_col;
        m1_local[thread_row][thread_col] = (row < M && col_k < K) ? m1.at(batch, row, col_k) : 0;

        uint row_k = k + thread_row;
        m2_local[thread_row][thread_col] = (row_k < K && col < N) ? m2.at(batch, row_k, col) : 0;

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint j = 0; j < BlockSize; j += TileSize) {
            metal::simdgroup_load(m1_simd, SimdData(m1_local), BlockSize, ulong2(j, simd_row));
            metal::simdgroup_load(m2_simd, SimdData(m2_local), BlockSize, ulong2(simd_col, j));
            metal::simdgroup_multiply_accumulate(mm_simd, m1_simd, m2_simd, mm_simd);
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Unload the result to a unused threadgroup matrix used to cache values from the
    // inputs matrices and then store the result into the output device memory.
    metal::simdgroup_store(mm_simd, SimdData(m1_local), BlockSize, ulong2(simd_col, simd_row));
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (row < M && col < N) {
        params.output.at(batch, row, col) = m1_local[thread_row][thread_col];
    }
}


__lib_metalchat_kernel3_tiled(gemm, 8, bfloat);
__lib_metalchat_kernel3_tiled(gemm, 16, bfloat);
__lib_metalchat_kernel3_tiled(gemm, 32, bfloat);
__lib_metalchat_kernel3_tiled(gemm, 8, float);
__lib_metalchat_kernel3_tiled(gemm, 16, float);
__lib_metalchat_kernel3_tiled(gemm, 32, float);


template <typename T> struct __gemv_parameters {
    tensor2<T> output;
    tensor2<const T> vec;
    tensor3<const T> mat;
    constant uint& block_size;
};


/// Vector multiplication vec(b x K) @ mat(b x K x N) -> C(b x N)
template <typename T, uint SimdSize = 32>
kernel void
gemv(
    __gemv_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]],
    uint simd_tid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
)
{
    const uint K = params.vec.size(1);
    const uint N = params.mat.size(2);

    float threadlocal_sum = 0.0f;

    const uint batch = gid.y;
    const uint n = gid.x;

    const uint begin = tid.x * params.block_size;
    const uint end = begin + params.block_size;

    for (uint k = begin; k < end && k < K && n < N; k++) {
        threadlocal_sum += params.vec.at(batch, k) * params.mat.at(batch, k, n);
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_total_sum[1];
    threadgroup float threadgroup_sum[SimdSize];

    if (simd_tid < SimdSize) {
        threadgroup_sum[simd_tid] = 0;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (simd_tid == 0) {
        threadgroup_sum[simd_gid] = acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        acc = metal::simd_sum(threadgroup_sum[simd_tid]);
        if (simd_tid == 0) {
            threadgroup_total_sum[0] = acc;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (n < N && tid.x == 0) {
        params.output.at(batch, n) = T(threadgroup_total_sum[0]);
    }
}


template [[host_name("gemv_bfloat")]]
kernel void gemv<bfloat>(__gemv_parameters<bfloat>, uint2, uint2, uint2, uint, uint);

template [[host_name("gemv_float")]]
kernel void gemv<float>(__gemv_parameters<float>, uint2, uint2, uint2, uint, uint);

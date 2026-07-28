// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __gemm_parameters {
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
gemm(
    __gemm_parameters<T> params,
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]
)
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

    float partial = float(0);

    uint r1 = block_row + thread_row;
    uint c2 = block_col + thread_col;

    for (uint k = 0; k < K; k += BlockSize) {
        uint c1 = k + thread_col;
        m1_local[thread_row][thread_col] = (r1 < M && c1 < K) ? m1.at(batch, r1, c1) : float(0);

        uint r2 = k + thread_row;
        m2_local[thread_row][thread_col] = (r2 < K && c2 < N) ? m2.at(batch, r2, c2) : float(0);

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


__lib_metalchat_kernel3_tiled(gemm, 8, bfloat);
__lib_metalchat_kernel3_tiled(gemm, 8, float);


template <typename T> struct __gemv_parameters {
    constant layout2& output_layout;
    device T* output;
    constant layout2& vec1_layout;
    device const T* vec1;
    constant layout3& mat2_layout;
    device const T* mat2;
    constant uint& block_size;
};


/// Vector multiplication vec1(b x K) @ mat2(b x K x N) -> C(b x N)
template <typename T>
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
    tensor2<T> out(params.output_layout, params.output);
    tensor2<const T> m1(params.vec1_layout, params.vec1);
    tensor3<const T> m2(params.mat2_layout, params.mat2);

    const uint K = m1.size(1);
    const uint N = m2.size(2);

    constexpr uint SimdSize = 32;
    float threadlocal_sum = 0.0f;

    const uint batch = gid.y;
    const uint n = gid.x;

    const uint begin = tid.x * params.block_size;
    const uint end = begin + params.block_size;

    for (uint k = begin; k < end && k < K && n < N; k++) {
        threadlocal_sum += m1.at(batch, k) * m2.at(batch, k, n);
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
        out.at(batch, n) = T(threadgroup_total_sum[0]);
    }
}


template [[host_name("gemv_bfloat")]]
kernel void gemv<bfloat>(__gemv_parameters<bfloat>, uint2, uint2, uint2, uint, uint);

template [[host_name("gemv_float")]]
kernel void gemv<float>(__gemv_parameters<float>, uint2, uint2, uint2, uint, uint);

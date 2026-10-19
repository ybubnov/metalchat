// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


template <typename T> struct __rmsnorm_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input);
    StructuredBuffer(T, 1, weight);
    constant float& eps;
    constant float& mu;
    constant uint& block_size;
};


template <typename T>
kernel void
rmsnorm(
    __rmsnorm_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_tid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
)
{
    constexpr uint SIMD_SIZE = 32;

    Tensor(params, output);
    Tensor(params, input);
    Tensor(params, weight);

    float threadlocal_sum = 0.0f;

    const uint dim_size = input.size(1);
    const uint i = gid;

    const uint begin = tid * params.block_size;
    const uint end = begin + params.block_size;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = input.at(i, j);
        threadlocal_sum += xj * xj;
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_inv_mean[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    //  Initialize shared memory
    if (simd_tid < SIMD_SIZE) {
        threadgroup_sum[simd_tid] = 0.0f;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write simd accumulations into shared memory
    if (simd_tid == 0) {
        threadgroup_sum[simd_gid] = acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Accumulate over simd groups
    if (simd_gid == 0) {
        acc = metal::simd_sum(threadgroup_sum[simd_tid]);
        if (simd_tid == 0) {
            const float mean_sq = (acc / dim_size);
            threadgroup_inv_mean[0] = metal::precise::rsqrt(mean_sq + params.eps);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write the outputs
    for (uint j = begin; j < end && j < dim_size; j++) {
        const float x = input.at(i, j);
        const float w = params.mu + weight.at(j);
        output.at(i, j) = T(w * x * threadgroup_inv_mean[0]);
    }
}


template [[host_name("rmsnorm_bfloat")]]
kernel void rmsnorm<bfloat>(__rmsnorm_parameters<bfloat>, uint, uint, uint, uint, uint);

template [[host_name("rmsnorm_float")]]
kernel void rmsnorm<float>(__rmsnorm_parameters<float>, uint, uint, uint, uint, uint);

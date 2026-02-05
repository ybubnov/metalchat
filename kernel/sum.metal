// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


template <typename T> struct __sum_parameters {
    tensor1<T> output;
    tensor2<const T> input;
    constant uint& block_size;
};


template <typename T>
kernel void
sum(__sum_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_tid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
    constexpr uint SIMD_SIZE = 32;

    float threadlocal_sum = 0.0f;

    const uint dim_size = params.input.size(1);
    const uint i = gid;

    const uint begin = tid * params.block_size;
    const uint end = begin + params.block_size;

    for (uint j = begin; j < end && j < dim_size; j++) {
        threadlocal_sum += params.input.at(i, j);
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_total_sum[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    if (simd_gid == 0) {
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

    if (tid == 0) {
        params.output.at(i) = T(threadgroup_total_sum[0]);
    }
}


template [[host_name("sum_bfloat")]]
kernel void sum<bfloat>(__sum_parameters<bfloat>, uint, uint, uint, uint);

template [[host_name("sum_float")]]
kernel void sum<float>(__sum_parameters<float>, uint, uint, uint, uint);

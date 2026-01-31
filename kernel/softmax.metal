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


template <typename T> struct __softmax_parameters {
    tensor2<T> output;
    tensor2<const T> input;
    constant uint& block_size;
};


template <typename T>
kernel void
softmax(
    __softmax_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_tid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
)
{
    constexpr uint SIMD_SIZE = 32;

    float threadlocal_sum = 0.0f;

    const uint dim_size = params.input.size(1);
    const uint i = gid;

    const uint begin = tid * params.block_size;
    const uint end = begin + params.block_size;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = float(params.input.at(i, j));
        threadlocal_sum += metal::fast::exp(xj);
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_exp_sum[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    //  Initialize shared memory
    if (simd_gid == 0) {
        threadgroup_sum[simd_tid] = 0;
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
            threadgroup_exp_sum[0] = 1 / acc;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write the outputs
    T exp_sum = T(threadgroup_exp_sum[0]);
    for (uint j = begin; j < end && j < dim_size; j++) {
        params.output.at(i, j) = T(exp(params.input.at(i, j))) * exp_sum;
    }
}


template [[host_name("softmax_bfloat")]]
kernel void softmax<bfloat>(__softmax_parameters<bfloat>, uint, uint, uint, uint);

template [[host_name("softmax_float")]]
kernel void softmax<float>(__softmax_parameters<float>, uint, uint, uint, uint);

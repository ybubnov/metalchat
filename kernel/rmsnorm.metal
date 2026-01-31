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
    constant layout2& output_layout;
    device T* output;
    constant layout2& input_layout;
    device const T* input;
    constant tensor_layout<1>& weight_layout;
    device const T* weight;
    constant float& eps;
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
    constexpr int SIMD_SIZE = 32;

    tensor2<const T> in(params.input_layout, params.input);
    tensor<const T, 1> w{params.weight, params.weight_layout};
    tensor2<T> out(params.output_layout, params.output);

    float threadlocal_sum = 0.0f;

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * params.block_size;
    const uint end = begin + params.block_size;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = float(in.at(i, j));
        threadlocal_sum += xj * xj;
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_inv_mean[1];
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
            threadgroup_inv_mean[0] = metal::fast::rsqrt((acc / dim_size) + params.eps);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write the outputs
    for (uint j = begin; j < end && j < dim_size; j++) {
        out.at(i, j) = w.at(j) * T(in.at(i, j) * threadgroup_inv_mean[0]);
    }
}


template [[host_name("rmsnorm_bfloat")]]
kernel void rmsnorm<bfloat>(__rmsnorm_parameters<bfloat>, uint, uint, uint, uint, uint);

template [[host_name("rmsnorm_float")]]
kernel void rmsnorm<float>(__rmsnorm_parameters<float>, uint, uint, uint, uint, uint);

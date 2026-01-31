// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T>
inline void
__swap(device T& a, device T& b)
{
    T __t = a;
    a = b;
    b = __t;
}


template <typename T> struct __sort_parameters {
    tensor2<T> values;
    tensor2<int32_t> indices;
    tensor2<const T> input;
    constant uint& block_size;
};


template <typename T>
kernel void
sort(
    __sort_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint dim_size_aligned = params.values.size(1);

    const uint batch = gid;
    const uint begin = tid * params.block_size;
    const uint end = begin + params.block_size;

    for (uint k = begin; k < end; k++) {
        if (k < dim_size) {
            params.values.at(batch, k) = params.input.at(batch, k);
        } else {
            params.values.at(batch, k) = T(-INFINITY);
        }
        params.indices.at(batch, k) = k;
    }

    // k is doubled every iteration
    for (uint k = 2; k <= dim_size_aligned; k = k * 2) {
        // j is halved at every iteration, with truncation of fractional parts
        for (uint j = k >> 1; j > 0; j = j >> 1) {
            threadgroup_barrier(metal::mem_flags::mem_device);

#pragma unroll
            for (uint i = begin; i < end; i++) {
                uint ij = i ^ j;

                device T& value_i = params.values.at(batch, i);
                device T& value_ij = params.values.at(batch, ij);

                if (i < ij) {
                    if (((i & k) == 0) && (value_i < value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(params.indices.at(batch, i), params.indices.at(batch, ij));
                    }
                    if (((i & k) != 0) && (value_i > value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(params.indices.at(batch, i), params.indices.at(batch, ij));
                    }
                }
            }
        }
    }
}


__lib_metalchat_kernel(sort, bfloat);
__lib_metalchat_kernel(sort, float);

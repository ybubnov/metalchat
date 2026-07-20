// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
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


template <typename T>
uint
__binary_search(thread tensor1<const T> data, T value, bool right)
{
    uint low = 0;
    uint high = data.size(0);

    while (low < high) {
        uint mid = __mean(low, high);
        T value_m = data.at(mid);

        if ((right && value_m <= value) || (!right && value_m < value)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return high;
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


template <typename T> struct __bucketize_parameters {
    tensor2<int32_t> output;
    tensor2<const T> input;
    tensor1<const T> boundaries;
    constant bool& right;
};


/// Returns the indices of the buckets to which each value in the input belongs, where the
/// boundaries of the buckets are set by boundaries.
template <typename T>
kernel void
bucketize(
    __bucketize_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint row_size = params.output.size(0);
    const uint dim_size = params.output.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        T input = params.input.at(i, k);
        params.output.at(i, k) = __binary_search(params.boundaries, input, /*right=*/params.right);
    }
}


__lib_metalchat_kernel2(bucketize, bfloat);
__lib_metalchat_kernel2(bucketize, float);

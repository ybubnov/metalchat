// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __cumsum_parameters {
    constant layout2& output_layout;
    device T* output;
    constant layout2& input_layout;
    device const T* input;
};


template <typename T, uint BlockSize>
kernel void
cumsum(
    __cumsum_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
)
{
    constexpr uint MaxBlocks = 256;

    tensor2<const T> in(params.input_layout, params.input);
    tensor2<T> out(params.output_layout, params.output);

    const uint row_size = in.size(0);
    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;
    const uint block_size = end > dim_size ? dim_size % BlockSize : BlockSize;

    threadgroup T group_sums[MaxBlocks];
    T local_sums[BlockSize];

    for (uint k = begin, j = 0; k < end && k < dim_size; k++, j++) {
        if (i < row_size) {
            if (j > 0) {
                local_sums[j] = in.at(i, k) + local_sums[j - 1];
            } else {
                local_sums[j] = in.at(i, k);
            }
        }
    }

    group_sums[tid] = local_sums[block_size - 1];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint active_thread = 1; active_thread < threadgroup_size; active_thread++) {
        if (tid >= active_thread) {
            T accumulated = group_sums[tid - active_thread];

            for (uint j = 0; j < block_size; j++) {
                local_sums[j] += accumulated;
            }
        }
    }

    for (uint k = begin; k < end && k < dim_size; k++) {
        if (i < row_size) {
            out.at(i, k) = local_sums[k - begin];
        }
    }
}


__lib_metalchat_kernel_tiled(cumsum, 2, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 4, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 8, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 16, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 32, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 64, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 128, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 256, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 512, bfloat);
__lib_metalchat_kernel_tiled(cumsum, 1024, bfloat);

__lib_metalchat_kernel_tiled(cumsum, 2, float);
__lib_metalchat_kernel_tiled(cumsum, 4, float);
__lib_metalchat_kernel_tiled(cumsum, 8, float);
__lib_metalchat_kernel_tiled(cumsum, 16, float);
__lib_metalchat_kernel_tiled(cumsum, 32, float);
__lib_metalchat_kernel_tiled(cumsum, 64, float);
__lib_metalchat_kernel_tiled(cumsum, 128, float);
__lib_metalchat_kernel_tiled(cumsum, 256, float);
__lib_metalchat_kernel_tiled(cumsum, 512, float);
__lib_metalchat_kernel_tiled(cumsum, 1024, float);

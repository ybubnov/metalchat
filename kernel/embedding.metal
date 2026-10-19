// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __embedding_parameters {
    RWStructuredBuffer(T, 3, output);
    StructuredBuffer(int32_t, 2, input);
    StructuredBuffer(T, 2, weight);
    constant uint& block_size;
};


/// Batched implementation of the embedding operation.
///
/// The kernel expects `input` to be 2-dimensional tensor, where the first dimension is a batch
/// dimension, and the second dimension represents a position of an embedding in the tensor
/// `weight`.
///
/// Essentially, the algorithm implements operation as presented in the code block below:
///
/// .. code-block:: c++
///
///     output[i][j][k] <- weight[input[i][j]][k]
///
template <typename T>
kernel void
embedding(
    __embedding_parameters<T> params,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, input);
    Tensor(params, weight);

    const uint dim_size = input.size(1);
    const uint emb_size = weight.size(1);
    const uint i = gid.z;

    const uint begin = gid.x * threadgroup_size.x + tid.x * params.block_size;
    const uint end = begin + params.block_size;

    const uint k = gid.y * threadgroup_size.y + tid.y;

    if (k < emb_size) {
#pragma unroll
        for (uint j = begin; j < end && j < dim_size; j++) {
            output.at(i, j, k) = weight.at(input.at(i, j), k);
        }
    }
}


__lib_metalchat_kernel3(embedding, bfloat);
__lib_metalchat_kernel3(embedding, float);

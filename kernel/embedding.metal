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
    constant layout3& output_layout;
    device T* output;
    constant layout2& input_layout;
    device const int32_t* input;
    constant layout2& weight_layout;
    device const T* weight;
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
    tensor2<const int32_t> in(params.input_layout, params.input);
    tensor2<const T> w(params.weight_layout, params.weight);
    tensor3<T> out(params.output_layout, params.output);

    const uint dim_size = in.size(1);
    const uint emb_size = w.size(1);
    const uint i = gid.z;

    const uint begin = gid.x * threadgroup_size.x + tid.x * params.block_size;
    const uint end = begin + params.block_size;

    const uint k = gid.y * threadgroup_size.y + tid.y;

    if (k < emb_size) {
#pragma unroll
        for (uint j = begin; j < end && j < dim_size; j++) {
            out.at(i, j, k) = w.at(in.at(i, j), k);
        }
    }
}


__lib_metalchat_kernel3(embedding, bfloat);
__lib_metalchat_kernel3(embedding, float);

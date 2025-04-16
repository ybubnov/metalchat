// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


#define __embedding_parameters(T)                               \
    constant tensor_layout<3>& output_layout     [[buffer(0)]], \
    device T* output                             [[buffer(1)]], \
    constant tensor_layout<2>& input_layout      [[buffer(2)]], \
    device const int32_t* input                  [[buffer(3)]], \
    constant tensor_layout<2>& weight_layout     [[buffer(4)]], \
    device const T* weight                       [[buffer(5)]], \
    uint3 gid [[threadgroup_position_in_grid]],                 \
    uint3 tid [[thread_position_in_threadgroup]],               \
    uint3 threadgroup_size [[threads_per_threadgroup]]


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
template <typename T, uint BlockSize, uint EmbeddingBlockSize>
kernel void
embedding(__embedding_parameters(T))
{
    tensor<const int32_t, 2> in{input, input_layout};
    tensor<const T, 2> w{weight, weight_layout};
    tensor<T, 3> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint emb_size = w.size(1);
    const uint i = gid.x;

    const uint j = tid.x + gid.z * threadgroup_size.z;

    const uint emb_begin = tid.y * EmbeddingBlockSize;
    const uint emb_end = emb_begin + EmbeddingBlockSize;

    if (j < dim_size) {
        for (uint k = emb_begin; k < emb_end && k < emb_size; k++) {
            out.at(i, j, k) = w.at(in.at(i, j), k);
        }
    }
}


template [[host_name("embedding_4x64_bfloat")]]
kernel void embedding<bfloat, 4, 64>(__embedding_parameters(bfloat));

template [[host_name("embedding_4x128_bfloat")]]
kernel void embedding<bfloat, 4, 128>(__embedding_parameters(bfloat));

template [[host_name("embedding_16x64_bfloat")]]
kernel void embedding<bfloat, 16, 64>(__embedding_parameters(bfloat));

template [[host_name("embedding_16x128_bfloat")]]
kernel void embedding<bfloat, 16, 128>(__embedding_parameters(bfloat));


template [[host_name("embedding_4x64_float")]]
kernel void embedding<float, 4, 64>(__embedding_parameters(float));

template [[host_name("embedding_4x128_float")]]
kernel void embedding<float, 4, 128>(__embedding_parameters(float));

template [[host_name("embedding_16x64_float")]]
kernel void embedding<float, 16, 64>(__embedding_parameters(float));

template [[host_name("embedding_16x128_float")]]
kernel void embedding<float, 16, 128>(__embedding_parameters(float));

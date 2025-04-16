// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __embedding_parameters(T)                               \
    constant tensor_layout<3>& output_layout     [[buffer(0)]], \
    device T* output                             [[buffer(1)]], \
    constant tensor_layout<2>& input_layout      [[buffer(2)]], \
    device const int32_t* input                  [[buffer(3)]], \
    constant tensor_layout<2>& weight_layout     [[buffer(4)]], \
    device const T* weight                       [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]],                 \
    uint2 tid [[thread_position_in_threadgroup]]


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
embedding(__embedding_parameters(T))
{
    // TODO: move these parameters to a template.
    constexpr uint BLOCK_SIZE = 4;
    constexpr uint EMBEDDING_SIZE = 128;

    tensor<const int32_t, 2> in{input, input_layout};
    tensor<const T, 2> w{weight, weight_layout};
    tensor<T, 3> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint emb_size = w.size(1);
    const uint i = gid.x;

    const uint begin = tid.x * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    const uint emb_begin = tid.y * EMBEDDING_SIZE;
    const uint emb_end = emb_begin + EMBEDDING_SIZE;

    for (uint j = begin; j < end && j < dim_size; j++) {
        for (uint k = emb_begin; k < emb_end && k < emb_size; k++) {
            out.at(i, j, k) = w.at(in.at(i, j), k);
        }
    }
}


template [[host_name("embedding_bf16")]]
kernel void embedding<bfloat>(__embedding_parameters(bfloat));


template [[host_name("embedding_float")]]
kernel void embedding<float>(__embedding_parameters(float));

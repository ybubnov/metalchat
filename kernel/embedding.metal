// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


#define __embedding_parameters(T)                               \
    constant tensor_layout<3>& output_layout     [[buffer(0)]], \
    device T* output                             [[buffer(1)]], \
    constant layout2& input_layout               [[buffer(2)]], \
    device const int32_t* input                  [[buffer(3)]], \
    constant layout2& weight_layout              [[buffer(4)]], \
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
template <typename T, uint BlockSize>
kernel void
embedding(__embedding_parameters(T))
{
    tensor2<const int32_t> in(input_layout, input);
    tensor2<const T> w(weight_layout, weight);
    tensor<T, 3> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint emb_size = w.size(1);
    const uint i = gid.x;

    const uint begin = tid.x * BlockSize;
    const uint end = begin + BlockSize;

    const uint k = tid.y + gid.z * threadgroup_size.y;

    if (k < emb_size) {
        for (uint j = begin; j < end && j < dim_size; j++) {
            out.at(i, j, k) = w.at(in.at(i, j), k);
        }
    }
}


template [[host_name("embedding_4_bfloat")]]
kernel void embedding<bfloat, 4>(__embedding_parameters(bfloat));

template [[host_name("embedding_16_bfloat")]]
kernel void embedding<bfloat, 16>(__embedding_parameters(bfloat));


template [[host_name("embedding_4_float")]]
kernel void embedding<float, 4>(__embedding_parameters(float));

template [[host_name("embedding_16_float")]]
kernel void embedding<float, 16>(__embedding_parameters(float));

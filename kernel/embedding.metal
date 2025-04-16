// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


#define __embedding_parameters(T)                \
    constant uint& dim_size [[buffer(0)]],       \
    constant uint& emb_size [[buffer(1)]],       \
    device const int32_t* input [[buffer(2)]],   \
    device const T* weight [[buffer(3)]],        \
    device T* output [[buffer(4)]],              \
    uint2 gid [[threadgroup_position_in_grid]],  \
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

    // Jump to a particular row of the input, then using `i + j` iterator, iterate
    // over the potions of ids.
    device const int32_t* in = input + gid.x * dim_size;

    // Jump to a particular row (batch) of the output. The logic below essentially
    // copies portion of elements from the `weight` tensor within a single block.
    device T* out = output + gid.x * emb_size * dim_size;

    uint i = tid.x * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    uint e = tid.y * EMBEDDING_SIZE;
    uint emb_remainder_size = emb_size % EMBEDDING_SIZE;
    uint emb_block_size = e + EMBEDDING_SIZE > emb_size ? emb_remainder_size : EMBEDDING_SIZE;

    for (uint j = 0; j < block_size; j++) {
        for (uint k = 0; k < emb_block_size; k++) {
            out[(i + j) * emb_size + e + k] = weight[in[i + j] * emb_size + e + k];
        }
    }
}


template [[host_name("embedding_bf16")]]
kernel void embedding<bfloat>(__embedding_parameters(bfloat));


template [[host_name("embedding_float")]]
kernel void embedding<float>(__embedding_parameters(float));

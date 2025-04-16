// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


using namespace metal;


#define __bmm_parameters(T)                                   \
    constant uint& M [[buffer(0)]],                           \
    constant uint& N [[buffer(1)]],                           \
    constant uint& K [[buffer(2)]],                           \
    device const T* mat1 [[buffer(3)]],                       \
    device const T* mat2 [[buffer(4)]],                       \
    device T* output [[buffer(5)]],                           \
    uint2 group_id [[threadgroup_position_in_grid]],          \
    uint2 group_thread_id [[thread_position_in_threadgroup]]


/// Matrix multiplication mat1(MxK) @ mat2(KxN) -> C(MxN)
template <typename T>
kernel void
bmm(__bmm_parameters(T))
{
    constexpr uint BLOCK_SIZE = 1;

    const uint mat2_cols = N;
    const uint mat1_cols = K;

    // Block index
    const uint bx = group_id.x;
    const uint by = group_id.y;

    // Thread index
    const uint tx = group_thread_id.x;
    const uint ty = group_thread_id.y;

    // Index of the first sub-matrix of A processed by the block
    const uint mat1_begin = mat1_cols * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    const uint mat1_end = mat1_begin + mat1_cols - 1;
    // Step size used to iterate through the sub-matrices of A
    const uint mat1_stride = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const uint mat2_begin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    const uint mat2_stride = BLOCK_SIZE * mat2_cols;

    // output_partial is used to store the element of the block sub-matrix
    // that is computed by the thread
    T output_partial = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (uint a = mat1_begin, b = mat2_begin; a <= mat1_end; a += mat1_stride, b += mat2_stride) {
        // Declaration of the shared memory array mat1_partial used to store the sub-matrix of A.
        threadgroup T mat1_partial[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array mat2_partial used to store the sub-matrix of B
        threadgroup T mat2_partial[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        mat1_partial[ty][tx] = mat1[a + mat1_cols * ty + tx];
        mat2_partial[ty][tx] = mat2[b + mat2_cols * ty + tx];

        // Synchronize to make sure the matrices are loaded
        threadgroup_barrier(mem_flags::mem_none);

        // Multiply the two matrices together; each thread computes one element
        // of the block sub-matrix
        for (uint k = 0; k < BLOCK_SIZE; ++k) {
            output_partial += mat1_partial[ty][k] * mat2_partial[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before
        // loading two new sub-matrices of A and B in the next iteration
        threadgroup_barrier(mem_flags::mem_none);
    }

    const uint offset = mat2_cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    output[offset + mat2_cols * ty + tx] = output_partial;
}


template [[host_name("bmm_bf16")]]
kernel void bmm<bfloat>(__bmm_parameters(bfloat));


template [[host_name("bmm_float")]]
kernel void bmm<float>(__bmm_parameters(float));

// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __bmm_parameters {
    constant tensor_layout<3>& output_layout;
    device T* output;
    constant tensor_layout<3>& mat1_layout;
    device const T* mat1;
    constant tensor_layout<3>& mat2_layout;
    device const T* mat2;
};


/// Matrix multiplication mat1(b x M x K) @ mat2(b x K x N) -> C(b x M x N)
template <typename T, uint BlockSize>
kernel void
bmm(__bmm_parameters<T> params,
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]])
{
    tensor<const T, 3> m1{params.mat1, params.mat1_layout};
    tensor<const T, 3> m2{params.mat2, params.mat2_layout};
    tensor<T, 3> out{params.output, params.output_layout};

    const uint M = m1.size(1);
    const uint K = m1.size(2);
    const uint N = m2.size(2);

    threadgroup T m1_local[BlockSize][BlockSize];
    threadgroup T m2_local[BlockSize][BlockSize];

    const uint block_row = group_id.x * BlockSize;
    const uint block_col = group_id.y * BlockSize;

    const uint batch = group_id.z;
    const uint thread_row = thread_id.x;
    const uint thread_col = thread_id.y;

    float partial = 0.0;

    for (uint k = 0; k < K; k += BlockSize) {
        uint r1 = block_row + thread_row;
        uint c1 = k + thread_col;
        if (r1 < M && c1 < K) {
            m1_local[thread_row][thread_col] = m1.at(batch, r1, c1);
        } else {
            m1_local[thread_row][thread_col] = T(0.0);
        }

        uint r2 = k + thread_row;
        uint c2 = block_col + thread_col;
        if (r2 < K && c2 < N) {
            m2_local[thread_row][thread_col] = m2.at(batch, r2, c2);
        } else {
            m2_local[thread_row][thread_col] = T(0.0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = 0; b < BlockSize; b++) {
            partial += m1_local[thread_row][b] * m2_local[b][thread_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint row = block_row + thread_row;
    uint col = block_col + thread_col;

    if (row < M && col < N) {
        out.at(batch, row, col) = T(partial);
    }
}


__lib_metalchat_kernel3(bmm, bfloat, 8);
__lib_metalchat_kernel3(bmm, bfloat, 16);
__lib_metalchat_kernel3(bmm, bfloat, 32);

__lib_metalchat_kernel3(bmm, float, 8);
__lib_metalchat_kernel3(bmm, float, 16);
__lib_metalchat_kernel3(bmm, float, 32);

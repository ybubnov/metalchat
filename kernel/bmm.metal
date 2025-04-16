// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __bmm_parameters {
    constant layout3& output_layout;
    device T* output;
    constant layout3& mat1_layout;
    device const T* mat1;
    constant layout3& mat2_layout;
    device const T* mat2;
};


/// Matrix multiplication mat1(b x M x K) @ mat2(b x K x N) -> C(b x M x N)
template <typename T, uint BlockSize>
kernel void
bmm(__bmm_parameters<T> params,
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]])
{
    tensor3<const T> m1(params.mat1_layout, params.mat1);
    tensor3<const T> m2(params.mat2_layout, params.mat2);
    tensor3<T> out(params.output_layout, params.output);

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

    T partial = T(0.0);

    uint r1 = metal::min(block_row + thread_row, M - 1);
    uint c2 = metal::min(block_col + thread_col, N - 1);

    for (uint k = 0; k < K; k += BlockSize) {
        uint c1 = k + thread_col;
        if (c1 < K) {
            m1_local[thread_row][thread_col] = m1.at(batch, r1, c1);
        } else {
            m1_local[thread_row][thread_col] = T(0.0);
        }

        uint r2 = k + thread_row;
        if (r2 < K) {
            m2_local[thread_row][thread_col] = m2.at(batch, r2, c2);
        } else {
            m2_local[thread_row][thread_col] = T(0.0);
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        partial += m1_local[thread_row][0] * m2_local[0][thread_col];
        partial += m1_local[thread_row][1] * m2_local[1][thread_col];
        partial += m1_local[thread_row][2] * m2_local[2][thread_col];
        partial += m1_local[thread_row][3] * m2_local[3][thread_col];
        partial += m1_local[thread_row][4] * m2_local[4][thread_col];
        partial += m1_local[thread_row][5] * m2_local[5][thread_col];
        partial += m1_local[thread_row][6] * m2_local[6][thread_col];
        partial += m1_local[thread_row][7] * m2_local[7][thread_col];
    }

    uint row = block_row + thread_row;
    uint col = block_col + thread_col;

    if (row < M && col < N) {
        out.at(batch, row, col) = partial;
    }
}


__lib_metalchat_kernel3(bmm, bfloat, 8);
__lib_metalchat_kernel3(bmm, float, 8);

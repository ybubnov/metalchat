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

    constexpr uint BM = 64;
    constexpr uint BN = 64;
    constexpr uint BK = 8;
    constexpr uint TM = 8;

    const uint block_row = group_id.y * BM;
    const uint block_col = group_id.x * BN;

    const uint batch = group_id.z;
    const uint thread_col = thread_id.x % BN;
    const uint thread_row = thread_id.x / BN;

    threadgroup T tile1[BM * BK];
    threadgroup T tile2[BK * BN];

    const uint inner_c1 = thread_id.x % BK;
    const uint inner_r1 = thread_id.x / BK;
    const uint inner_c2 = thread_id.x % BN;
    const uint inner_r2 = thread_id.x / BN;

    float partial[TM] = {0.0};

    for (uint k = 0; k < K; k += BK) {
        uint r1 = inner_r1 + block_row;
        uint c1 = inner_c1 + k;

        if (r1 < M && c1 < K) {
            tile1[inner_r1 * BK + inner_c1] = m1.at(batch, r1, c1);
        } else {
            tile1[inner_r1 * BK + inner_c1] = T(0.0);
        }

        uint r2 = inner_r2 + k;
        uint c2 = inner_c2 + block_col;

        if (r2 < K && c2 < N) {
            tile2[inner_r2 * BN + inner_c2] = m2.at(batch, r2, c2);
        } else {
            tile2[inner_r2 * BN + inner_c2] = T(0.0);
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        for (uint b = 0; b < BK; b++) {
            const float partial2 = tile2[b * BN + thread_col];

            for (uint i = 0; i < TM; i++) {
                partial[i] += tile1[(thread_row * TM + i) * BK + b] * partial2;
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    const uint col = block_col + thread_col;

    for (uint i = 0; i < TM; i++) {
        const uint row = block_row + thread_row * TM + i;
        if (row < M && col < N) {
            out.at(batch, row, col) = T(partial[i]);
        }
    }
}


__lib_metalchat_kernel3(bmm, bfloat, 8);
__lib_metalchat_kernel3(bmm, float, 8);

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

    constexpr uint BM = 16;
    constexpr uint BN = 16;
    constexpr uint BK = 4;
    constexpr uint TM = 4;
    constexpr uint TN = 4;

    const uint block_row = group_id.y * BM;
    const uint block_col = group_id.x * BN;

    const uint batch = group_id.z;
    const uint thread_row = thread_id.x / (BN / TN);
    const uint thread_col = thread_id.x % (BN / TN);
    const uint thread_size = (BM * BN) / (TM * TN);

    threadgroup T tile1[BM][BK];
    threadgroup T tile2[BK][BN];

    const uint inner_r1 = thread_id.x / BK;
    const uint inner_c1 = thread_id.x % BK;
    const uint stride1 = thread_size / BK;

    const uint inner_r2 = thread_id.x / BN;
    const uint inner_c2 = thread_id.x % BN;
    const uint stride2 = thread_size / BN;

    // T partial[TM][TN] = {};
    metal::float4x4 partial = {};
    T cache_m[TM] = {};
    T cache_n[TN] = {};

    for (uint k = 0; k < K; k += BK) {

#pragma unroll(BM)
        for (uint off = 0; off < BM; off += stride1) {
            uint r1 = inner_r1 + block_row + off;
            uint c1 = inner_c1 + k;

            if (r1 < M && c1 < K) {
                tile1[inner_r1 + off][inner_c1] = m1.at(batch, r1, c1);
            } else {
                tile1[inner_r1 + off][inner_c1] = T(0.0);
            }
        }

#pragma unroll(BK)
        for (uint off = 0; off < BK; off += stride2) {
            uint r2 = inner_r2 + k + off;
            uint c2 = inner_c2 + block_col;

            if (r2 < K && c2 < N) {
                tile2[inner_r2 + off][inner_c2] = m2.at(batch, r2, c2);
            } else {
                tile2[inner_r2 + off][inner_c2] = T(0.0);
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#pragma unroll(BK)
        for (uint b = 0; b < BK; b++) {
            for (uint i = 0; i < TM; i++) {
                cache_m[i] = tile1[thread_row * TM + i][b];
            }

            for (uint j = 0; j < TN; j++) {
                cache_n[j] = tile2[b][thread_col * TN + j];
            }

            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    partial[i][j] += cache_m[i] * cache_n[j];
                }
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }


    for (uint i = 0; i < TM; i++) {
        const uint row = block_row + thread_row * TM + i;

        for (uint j = 0; j < TN; j++) {
            const uint col = block_col + thread_col * TN + j;

            if (row < M && col < N) {
                out.at(batch, row, col) = T(partial[i][j]);
                // out.at(batch, row, col) = T(3.0);
            }
        }
    }
}


__lib_metalchat_kernel3(bmm, bfloat, 8);
__lib_metalchat_kernel3(bmm, float, 8);

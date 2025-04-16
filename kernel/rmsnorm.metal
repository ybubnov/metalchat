// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


#define __rmsnorm_parameters(T)                        \
    constant uint& dim_size [[buffer(0)]],             \
    device const T* input [[buffer(1)]],               \
    device const T* weight [[buffer(2)]],              \
    constant float& eps [[buffer(3)]],                 \
    device T* output [[buffer(4)]],                    \
    uint gid [[threadgroup_position_in_grid]],         \
    uint tid [[thread_index_in_threadgroup]],          \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint simd_tid [[thread_index_in_simdgroup]],       \
    uint simd_gid [[simdgroup_index_in_threadgroup]]


template <typename T>
kernel void
rmsnorm(__rmsnorm_parameters(T))
{
    constexpr int SIMD_SIZE = 32;
    constexpr int BLOCK_SIZE = 4;

    device const T* in = input + gid * dim_size;
    device T* out = output + gid * dim_size;

    float threadlocal_sum = 0.0f;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        float xj = in[i + j];
        threadlocal_sum += xj * xj;
    }

    float acc = simd_sum(threadlocal_sum);

    threadgroup float threadgroup_inv_mean[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    //  Initialize shared memory
    if (simd_gid == 0) {
        threadgroup_sum[simd_tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write simd accumulations into shared memory
    if (simd_tid == 0) {
        threadgroup_sum[simd_gid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate over simd groups
    if (simd_gid == 0) {
        acc = simd_sum(threadgroup_sum[simd_tid]);
        if (simd_tid == 0) {
            threadgroup_inv_mean[0] = metal::fast::rsqrt((acc / dim_size) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the outputs
    for (uint j = 0; j < block_size; j++) {
        out[i + j] = weight[i + j] * T(in[i + j] * threadgroup_inv_mean[0]);
    }
}


template [[host_name("rmsnorm_bf16")]]
kernel void rmsnorm<bfloat>(__rmsnorm_parameters(bfloat));


template [[host_name("rmsnorm_float")]]
kernel void rmsnorm<float>(__rmsnorm_parameters(float));

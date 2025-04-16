// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


kernel void
embedding_f16(
    device const int32_t* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const int64_t* stride [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]
)
{
    output[index.x * stride[0] + index.y] = weight[input[index.x] * stride[0] + index.y];
}


kernel void
sum_f16(
    device const bfloat* input [[buffer(0)]],
    volatile device atomic_float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
)
{
    // TODO: conversion to float type does not produce correct sum.
    atomic_fetch_add_explicit(output, (float)input[index], memory_order_relaxed);
}


namespace detail {

/// Approximate inverse square root implementation.
bfloat
rsqrt(bfloat number)
{
    union {
        bfloat f;
        uint32_t i;
    } conv = {.f = number};

    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= bfloat(1.5f) - (number * bfloat(0.5f) * conv.f * conv.f);
    return conv.f;
}

} // namespace detail


kernel void
rmsnorm_f16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* eps [[buffer(2)]],
    device const int32_t* input_size_ [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_tid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
)
{
    constexpr int SIMD_SIZE = 32;
    constexpr int BLOCK_SIZE = 4;

    threadgroup float threadgroup_inv_mean[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    uint input_size = uint(input_size_[0]);
    bfloat threadlocal_sum = bfloat(0.0f);

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = input_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > input_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        bfloat xj = input[i + j];
        threadlocal_sum += xj * xj;
    }

    float acc = static_cast<float>(threadlocal_sum);
    acc = simd_sum(acc);

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
            threadgroup_inv_mean[0] = metal::fast::rsqrt(acc / input_size + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the outputs
    bfloat inv_mean = bfloat(threadgroup_inv_mean[0]);
    for (uint j = 0; j < block_size; j++) {
        output[i + j] = weight[i + j] * input[i + j] * inv_mean;
    }
}

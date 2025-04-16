// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


kernel void
rmsnorm_f16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    constant bfloat& eps [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
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
            threadgroup_inv_mean[0] = metal::fast::rsqrt(acc / input_size + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the outputs
    bfloat inv_mean = bfloat(threadgroup_inv_mean[0]);
    for (uint j = 0; j < block_size; j++) {
        output[i + j] = weight[i + j] * input[i + j] * inv_mean;
    }
}

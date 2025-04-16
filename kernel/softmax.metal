// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


using namespace metal;


#define __softmax_parameters(T)                                                                    \
    constant uint &dim_size [[buffer(0)]], device const T *input [[buffer(1)]],                    \
        device T *output [[buffer(2)]], uint gid [[threadgroup_position_in_grid]],                 \
        uint tid [[thread_index_in_threadgroup]], uint simd_tid [[thread_index_in_simdgroup]],     \
        uint simd_gid [[simdgroup_index_in_threadgroup]]


template <typename T>
kernel void
softmax(__softmax_parameters(T))
{
    constexpr uint SIMD_SIZE = 32;
    constexpr uint BLOCK_SIZE = 4;

    device const T* in = input + gid * dim_size;
    device T* out = output + gid * dim_size;

    float threadlocal_sum = 0.0f;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        T xj = in[i + j];
        threadlocal_sum += metal::fast::exp(xj);
    }

    float acc = simd_sum(threadlocal_sum);

    threadgroup float threadgroup_exp_sum[1];
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
            threadgroup_exp_sum[0] = 1 / acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the outputs
    T exp_sum = T(threadgroup_exp_sum[0]);
    for (uint j = 0; j < block_size; j++) {
        out[i + j] = T(exp(in[i + j])) * exp_sum;
    }
}


template [[host_name("softmax_bf16")]]
kernel void softmax<bfloat>(__softmax_parameters(bfloat));


template [[host_name("softmax_float")]]
kernel void softmax<float>(__softmax_parameters(float));

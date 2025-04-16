// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __rmsnorm_parameters(T)                             \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    constant tensor_layout<1>& weight_layout [[buffer(4)]], \
    device const T* weight                   [[buffer(5)]], \
    constant float& eps                      [[buffer(6)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_index_in_threadgroup]],               \
    uint threadgroup_size [[threads_per_threadgroup]],      \
    uint simd_tid [[thread_index_in_simdgroup]],            \
    uint simd_gid [[simdgroup_index_in_threadgroup]]


template <typename T>
kernel void
rmsnorm(__rmsnorm_parameters(T))
{
    constexpr int SIMD_SIZE = 32;
    constexpr int BLOCK_SIZE = 4;

    tensor<const T, 2> in{input, input_layout};
    tensor<const T, 1> w{weight, weight_layout};
    tensor<T, 2> out{output, output_layout};

    float threadlocal_sum = 0.0f;

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = float(in.at(i, j));
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
    for (uint j = begin; j < end && j < dim_size; j++) {
        out.at(i, j) = w.at(j) * T(in.at(i, j) * threadgroup_inv_mean[0]);
    }
}


template [[host_name("rmsnorm_bf16")]]
kernel void rmsnorm<bfloat>(__rmsnorm_parameters(bfloat));


template [[host_name("rmsnorm_float")]]
kernel void rmsnorm<float>(__rmsnorm_parameters(float));

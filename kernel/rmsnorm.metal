// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


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


template <typename T, uint BlockSize>
kernel void
rmsnorm(__rmsnorm_parameters(T))
{
    constexpr int SIMD_SIZE = 32;

    tensor<const T, 2> in{input, input_layout};
    tensor<const T, 1> w{weight, weight_layout};
    tensor<T, 2> out{output, output_layout};

    float threadlocal_sum = 0.0f;

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = float(in.at(i, j));
        threadlocal_sum += xj * xj;
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_inv_mean[1];
    threadgroup float threadgroup_sum[SIMD_SIZE];

    //  Initialize shared memory
    if (simd_gid == 0) {
        threadgroup_sum[simd_tid] = 0;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write simd accumulations into shared memory
    if (simd_tid == 0) {
        threadgroup_sum[simd_gid] = acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Accumulate over simd groups
    if (simd_gid == 0) {
        acc = metal::simd_sum(threadgroup_sum[simd_tid]);
        if (simd_tid == 0) {
            threadgroup_inv_mean[0] = metal::fast::rsqrt((acc / dim_size) + eps);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write the outputs
    for (uint j = begin; j < end && j < dim_size; j++) {
        out.at(i, j) = w.at(j) * T(in.at(i, j) * threadgroup_inv_mean[0]);
    }
}


template [[host_name("rmsnorm_1_bfloat")]]
kernel void rmsnorm<bfloat, 1>(__rmsnorm_parameters(bfloat));

template [[host_name("rmsnorm_2_bfloat")]]
kernel void rmsnorm<bfloat, 2>(__rmsnorm_parameters(bfloat));

template [[host_name("rmsnorm_4_bfloat")]]
kernel void rmsnorm<bfloat, 4>(__rmsnorm_parameters(bfloat));

template [[host_name("rmsnorm_8_bfloat")]]
kernel void rmsnorm<bfloat, 8>(__rmsnorm_parameters(bfloat));

template [[host_name("rmsnorm_16_bfloat")]]
kernel void rmsnorm<bfloat, 16>(__rmsnorm_parameters(bfloat));

template [[host_name("rmsnorm_32_bfloat")]]
kernel void rmsnorm<bfloat, 32>(__rmsnorm_parameters(bfloat));


template [[host_name("rmsnorm_1_float")]]
kernel void rmsnorm<float, 1>(__rmsnorm_parameters(float));

template [[host_name("rmsnorm_2_float")]]
kernel void rmsnorm<float, 2>(__rmsnorm_parameters(float));

template [[host_name("rmsnorm_4_float")]]
kernel void rmsnorm<float, 4>(__rmsnorm_parameters(float));

template [[host_name("rmsnorm_8_float")]]
kernel void rmsnorm<float, 8>(__rmsnorm_parameters(float));

template [[host_name("rmsnorm_16_float")]]
kernel void rmsnorm<float, 16>(__rmsnorm_parameters(float));

template [[host_name("rmsnorm_32_float")]]
kernel void rmsnorm<float, 32>(__rmsnorm_parameters(float));

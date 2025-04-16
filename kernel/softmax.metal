// vi: set filetype=cpp :

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __softmax_parameters(T)                             \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_index_in_threadgroup]],               \
    uint simd_tid [[thread_index_in_simdgroup]],            \
    uint simd_gid [[simdgroup_index_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
softmax(__softmax_parameters(T))
{
    constexpr uint SIMD_SIZE = 32;

    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    float threadlocal_sum = 0.0f;

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint j = begin; j < end && j < dim_size; j++) {
        float xj = float(in.at(i, j));
        threadlocal_sum += metal::fast::exp(xj);
    }

    float acc = metal::simd_sum(threadlocal_sum);

    threadgroup float threadgroup_exp_sum[1];
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
            threadgroup_exp_sum[0] = 1 / acc;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Write the outputs
    T exp_sum = T(threadgroup_exp_sum[0]);
    for (uint j = begin; j < end && j < dim_size; j++) {
        out.at(i, j) = T(exp(in.at(i, j))) * exp_sum;
    }
}


template [[host_name("softmax_1_bfloat")]]
kernel void softmax<bfloat, 1>(__softmax_parameters(bfloat));

template [[host_name("softmax_2_bfloat")]]
kernel void softmax<bfloat, 2>(__softmax_parameters(bfloat));

template [[host_name("softmax_4_bfloat")]]
kernel void softmax<bfloat, 4>(__softmax_parameters(bfloat));

template [[host_name("softmax_8_bfloat")]]
kernel void softmax<bfloat, 8>(__softmax_parameters(bfloat));

template [[host_name("softmax_16_bfloat")]]
kernel void softmax<bfloat, 16>(__softmax_parameters(bfloat));

template [[host_name("softmax_32_bfloat")]]
kernel void softmax<bfloat, 32>(__softmax_parameters(bfloat));

template [[host_name("softmax_128_bfloat")]]
kernel void softmax<bfloat, 128>(__softmax_parameters(bfloat));


template [[host_name("softmax_1_float")]]
kernel void softmax<float, 1>(__softmax_parameters(float));

template [[host_name("softmax_2_float")]]
kernel void softmax<float, 2>(__softmax_parameters(float));

template [[host_name("softmax_4_float")]]
kernel void softmax<float, 4>(__softmax_parameters(float));

template [[host_name("softmax_8_float")]]
kernel void softmax<float, 8>(__softmax_parameters(float));

template [[host_name("softmax_16_float")]]
kernel void softmax<float, 16>(__softmax_parameters(float));

template [[host_name("softmax_32_float")]]
kernel void softmax<float, 32>(__softmax_parameters(float));

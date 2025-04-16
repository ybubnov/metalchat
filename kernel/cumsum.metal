// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "tensor.h"


#define __cumsum_parameters(T)                              \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_position_in_threadgroup]],            \
    uint threadgroup_size [[threads_per_threadgroup]]


template <typename T, uint BlockSize, uint MaxBlocks = 1024>
kernel void
cumsum(__cumsum_parameters(T))
{
    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;
    const uint block_size = end > dim_size ? dim_size % BlockSize : BlockSize;

    threadgroup T group_sums[MaxBlocks];
    T local_sums[BlockSize];

    for (uint k = begin, j = 0; k < end && k < dim_size; k++, j++) {
        if (j > 0) {
            local_sums[j] = in.at(i, k) + local_sums[j - 1];
        } else {
            local_sums[j] = in.at(i, k);
        }
    }

    group_sums[tid] = local_sums[block_size - 1];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint active_thread = 1; active_thread < threadgroup_size; active_thread++) {
        if (tid >= active_thread) {
            T accumulated = group_sums[tid - active_thread];

            for (uint j = 0; j < block_size; j++) {
                local_sums[j] += accumulated;
            }
        }
    }

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(i, k) = local_sums[k - begin];
    }
}


template [[host_name("cumsum_1_bfloat")]]
kernel void cumsum<bfloat, 1>(__cumsum_parameters(bfloat));

template [[host_name("cumsum_4_bfloat")]]
kernel void cumsum<bfloat, 4>(__cumsum_parameters(bfloat));

template [[host_name("cumsum_16_bfloat")]]
kernel void cumsum<bfloat, 16>(__cumsum_parameters(bfloat));

template [[host_name("cumsum_32_bfloat")]]
kernel void cumsum<bfloat, 32>(__cumsum_parameters(bfloat));

template [[host_name("cumsum_256_bfloat")]]
kernel void cumsum<bfloat, 256>(__cumsum_parameters(bfloat));


template [[host_name("cumsum_1_float")]]
kernel void cumsum<float, 1>(__cumsum_parameters(float));

template [[host_name("cumsum_4_float")]]
kernel void cumsum<float, 4>(__cumsum_parameters(float));

template [[host_name("cumsum_16_float")]]
kernel void cumsum<float, 16>(__cumsum_parameters(float));

template [[host_name("cumsum_32_float")]]
kernel void cumsum<float, 32>(__cumsum_parameters(float));

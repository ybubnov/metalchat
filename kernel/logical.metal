// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


using namespace metal;


#define __gt_parameters(T)                                     \
    constant tensor_layout<2>& output_layout    [[buffer(0)]], \
    device bool* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input1_layout    [[buffer(2)]], \
    device const T* input1                      [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout    [[buffer(4)]], \
    device const T* input2                      [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],                 \
    uint tid [[thread_index_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
gt(__gt_parameters(T))
{
    tensor<const T, 2> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<bool, 2> out{output, output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) > in2.at(i, k);
    }
}


template [[host_name("gt_8_bf16")]]
kernel void gt<bfloat, 8>(__gt_parameters(bfloat));

template [[host_name("gt_16_bf16")]]
kernel void gt<bfloat, 16>(__gt_parameters(bfloat));

template [[host_name("gt_32_bf16")]]
kernel void gt<bfloat, 32>(__gt_parameters(bfloat));


template [[host_name("gt_8_float")]]
kernel void gt<float, 8>(__gt_parameters(float));

template [[host_name("gt_16_float")]]
kernel void gt<float, 16>(__gt_parameters(float));

template [[host_name("gt_32_float")]]
kernel void gt<float, 32>(__gt_parameters(float));

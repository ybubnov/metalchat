// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


using namespace metal;


#define __sum_parameters(T)                                    \
    constant tensor_layout<2>& output_layout    [[buffer(0)]], \
    device T* output                            [[buffer(1)]], \
    constant tensor_layout<2>& input1_layout    [[buffer(2)]], \
    device const T* input1                      [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout    [[buffer(4)]], \
    device const T* input2                      [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],                 \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
sum(__sum_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BLOCK_SIZE;
    const uint end = begin + BLOCK_SIZE;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) + in2.at(i, k);
    }
}


template [[host_name("sum_bf16")]]
kernel void sum<bfloat>(__sum_parameters(bfloat));


template [[host_name("sum_float")]]
kernel void sum<float>(__sum_parameters(float));


#define __sum2_parameters(T)                                \
    constant tensor_layout<3>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<3>& input1_layout [[buffer(2)]], \
    device const T* input1                   [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout [[buffer(4)]], \
    device const T* input2                   [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]],             \
    uint2 tid [[thread_position_in_threadgroup]]


template <typename T>
kernel void
sum2(__sum2_parameters(T))
{
    constexpr uint BLOCK_SIZE_X = 4;
    constexpr uint BLOCK_SIZE_Y = 4;

    tensor<const T, 3> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<T, 3> out{output, output_layout};

    const uint dim0_size = in2.size(0);
    const uint dim1_size = in2.size(1);
    const uint i = gid.x;

    const uint begin_x = tid.x * BLOCK_SIZE_X;
    const uint end_x = begin_x + BLOCK_SIZE_X;

    const uint begin_y = tid.y * BLOCK_SIZE_Y;
    const uint end_y = begin_y + BLOCK_SIZE_Y;

    for (uint j = begin_x; j < end_x && j < dim0_size; j++) {
        for (uint k = begin_y; k < end_y && k < dim1_size; k++) {
            out.at(i, j, k) = in1.at(i, j, k) + in2.at(j, k);
        }
    }
}


template [[host_name("sum2_bf16")]]
kernel void sum2<bfloat>(__sum2_parameters(bfloat));


template [[host_name("sum2_float")]]
kernel void sum2<float>(__sum2_parameters(float));

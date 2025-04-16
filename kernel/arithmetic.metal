// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


using namespace metal;


#define __add_parameters(T)                                    \
    constant tensor_layout<2>& output_layout    [[buffer(0)]], \
    device T* output                            [[buffer(1)]], \
    constant tensor_layout<2>& input1_layout    [[buffer(2)]], \
    device const T* input1                      [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout    [[buffer(4)]], \
    device const T* input2                      [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],                 \
    uint tid [[thread_index_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
add(__add_parameters(T))
{
    tensor<const T, 2> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) + in2.at(i, k);
    }
}


template [[host_name("add_8_bf16")]]
kernel void add<bfloat, 8>(__add_parameters(bfloat));

template [[host_name("add_16_bf16")]]
kernel void add<bfloat, 16>(__add_parameters(bfloat));

template [[host_name("add_32_bf16")]]
kernel void add<bfloat, 32>(__add_parameters(bfloat));


template [[host_name("add_8_float")]]
kernel void add<float, 8>(__add_parameters(float));

template [[host_name("add_16_float")]]
kernel void add<float, 16>(__add_parameters(float));

template [[host_name("add_32_float")]]
kernel void add<float, 32>(__add_parameters(float));


#define __add2_parameters(T)                                \
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
add2(__add2_parameters(T))
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


template [[host_name("add2_bf16")]]
kernel void add2<bfloat>(__add2_parameters(bfloat));


template [[host_name("add2_float")]]
kernel void add2<float>(__add2_parameters(float));


#define __sub_parameters(T)                                    \
    constant tensor_layout<2>& output_layout    [[buffer(0)]], \
    device T* output                            [[buffer(1)]], \
    constant tensor_layout<2>& input1_layout    [[buffer(2)]], \
    device const T* input1                      [[buffer(3)]], \
    constant tensor_layout<2>& input2_layout    [[buffer(4)]], \
    device const T* input2                      [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],                 \
    uint tid [[thread_index_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
sub(__sub_parameters(T))
{
    tensor<const T, 2> in1{input1, input1_layout};
    tensor<const T, 2> in2{input2, input2_layout};
    tensor<T, 2> out{output, output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) - in2.at(i, k);
    }
}


template [[host_name("sub_8_bf16")]]
kernel void sub<bfloat, 8>(__sub_parameters(bfloat));

template [[host_name("sub_16_bf16")]]
kernel void sub<bfloat, 16>(__sub_parameters(bfloat));

template [[host_name("sub_32_bf16")]]
kernel void sub<bfloat, 32>(__sub_parameters(bfloat));


template [[host_name("sub_8_float")]]
kernel void sub<float, 8>(__sub_parameters(float));

template [[host_name("sub_16_float")]]
kernel void sub<float, 16>(__sub_parameters(float));

template [[host_name("sub_32_float")]]
kernel void sub<float, 32>(__sub_parameters(float));

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


#define __sum2_parameters(T)                     \
    constant uint& dim0_size [[buffer(0)]],      \
    constant uint& dim1_size [[buffer(1)]],      \
    device const T* input1 [[buffer(2)]],        \
    device const T* input2 [[buffer(3)]],        \
    device T* output [[buffer(4)]],              \
    uint2 gid [[threadgroup_position_in_grid]],  \
    uint2 tid [[thread_position_in_threadgroup]]


template <typename T>
kernel void
sum2(__sum2_parameters(T))
{
    constexpr uint BLOCK_SIZE_X = 4;
    constexpr uint BLOCK_SIZE_Y = 4;

    device const T* in1 = input1 + gid.x * dim0_size * dim1_size;
    device T* out = output + gid.x * dim0_size * dim1_size;

    uint x = tid.x * BLOCK_SIZE_X;
    uint remainder_size_x = dim0_size % BLOCK_SIZE_X;
    uint block_size_x = x + BLOCK_SIZE_X > dim0_size ? remainder_size_x : BLOCK_SIZE_X;

    uint y = tid.y * BLOCK_SIZE_Y;
    uint remainder_size_y = dim1_size % BLOCK_SIZE_Y;
    uint block_size_y = y + BLOCK_SIZE_Y > dim1_size ? remainder_size_y : BLOCK_SIZE_Y;

    for (uint j = 0; j < block_size_x; j++) {
        for (uint k = 0; k < block_size_y; k++) {
            // out[(x + j) * dim1_size + y + k] = in1[(x + j) * dim1_size + y + k];
            uint index = (x + j) * dim1_size + y + k;
            out[index] = in1[index] + input2[index];
        }
    }
}


template [[host_name("sum2_bf16")]]
kernel void sum2<bfloat>(__sum2_parameters(bfloat));


template [[host_name("sum2_float")]]
kernel void sum2<float>(__sum2_parameters(float));

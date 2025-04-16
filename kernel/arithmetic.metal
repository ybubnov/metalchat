// vi: set filetype=cpp :

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __add_parameters {
    constant tensor_layout<2>& output_layout [[buffer(0)]];
    device T* output [[buffer(1)]];
    constant tensor_layout<2>& input1_layout [[buffer(2)]];
    device const T* input1 [[buffer(3)]];
    constant tensor_layout<2>& input2_layout [[buffer(4)]];
    device const T* input2 [[buffer(5)]];
};


template <typename T, uint BlockSize>
kernel void
add(__add_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    tensor<const T, 2> in1{params.input1, params.input1_layout};
    tensor<const T, 2> in2{params.input2, params.input2_layout};
    tensor<T, 2> out{params.output, params.output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        out.at(i, k) = in1.at(i, k) + in2.at(i, k);
    }
}


__lib_metalchat_kernel2x(add, bfloat, 8);
__lib_metalchat_kernel2x(add, bfloat, 16);
__lib_metalchat_kernel2x(add, bfloat, 32);

__lib_metalchat_kernel2x(add, float, 8);
__lib_metalchat_kernel2x(add, float, 16);
__lib_metalchat_kernel2x(add, float, 32);


template <typename T> struct __add2_parameters {
    constant tensor_layout<3>& output_layout;
    device T* output;
    constant tensor_layout<3>& input1_layout;
    device const T* input1;
    constant tensor_layout<2>& input2_layout;
    device const T* input2;
};


template <typename T, uint BlockSize>
kernel void
add2(
    __add2_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
)
{
    tensor<const T, 3> in1{params.input1, params.input1_layout};
    tensor<const T, 2> in2{params.input2, params.input2_layout};
    tensor<T, 3> out{params.output, params.output_layout};

    const uint dim0_size = in2.size(0);
    const uint dim1_size = in2.size(1);
    const uint i = gid.x;

    const uint begin_x = tid.x * BlockSize;
    const uint end_x = begin_x + BlockSize;

    const uint begin_y = tid.y * BlockSize;
    const uint end_y = begin_y + BlockSize;

    for (uint j = begin_x; j < end_x && j < dim0_size; j++) {
        for (uint k = begin_y; k < end_y && k < dim1_size; k++) {
            out.at(i, j, k) = in1.at(i, j, k) + in2.at(j, k);
        }
    }
}


__lib_metalchat_kernel2(add2, bfloat, 8);
__lib_metalchat_kernel2(add2, bfloat, 16);
__lib_metalchat_kernel2(add2, bfloat, 32);

__lib_metalchat_kernel2(add2, float, 8);
__lib_metalchat_kernel2(add2, float, 16);
__lib_metalchat_kernel2(add2, float, 32);


template <typename T> struct __sub_parameters {
    constant tensor_layout<2>& output_layout;
    device T* output;
    constant tensor_layout<2>& input1_layout;
    device const T* input1;
    constant tensor_layout<2>& input2_layout;
    device const T* input2;
};


template <typename T, uint BlockSize>
kernel void
sub(__sub_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    tensor<const T, 2> in1{params.input1, params.input1_layout};
    tensor<const T, 2> in2{params.input2, params.input2_layout};
    tensor<T, 2> out{params.output, params.output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        out.at(i, k) = in1.at(i, k) - in2.at(i, k);
    }
}


__lib_metalchat_kernel2x(sub, bfloat, 8);
__lib_metalchat_kernel2x(sub, bfloat, 16);
__lib_metalchat_kernel2x(sub, bfloat, 32);
__lib_metalchat_kernel2x(sub, bfloat, 128);

__lib_metalchat_kernel2x(sub, float, 8);
__lib_metalchat_kernel2x(sub, float, 16);
__lib_metalchat_kernel2x(sub, float, 32);

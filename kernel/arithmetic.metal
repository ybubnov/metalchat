// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __add_parameters {
    constant layout2& output_layout [[buffer(0)]];
    device T* output [[buffer(1)]];
    constant layout2& input1_layout [[buffer(2)]];
    device const T* input1 [[buffer(3)]];
    constant layout2& input2_layout [[buffer(4)]];
    device const T* input2 [[buffer(5)]];
};


template <typename T>
kernel void
add(__add_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    tensor2<const T> in1(params.input1_layout, params.input1);
    tensor2<const T> in2(params.input2_layout, params.input2);
    tensor2<T> out(params.output_layout, params.output);

    const uint dim_size = in1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (k < dim_size) {
        out.at(i, k) = in1.at(i, k) + in2.at(i, k);
    }
}


__lib_metalchat_kernel2(add, bfloat);
__lib_metalchat_kernel2(add, float);


template <typename T> struct __add2_parameters {
    constant layout3& output_layout;
    device T* output;
    constant layout3& input1_layout;
    device const T* input1;
    constant layout2& input2_layout;
    device const T* input2;
};


template <typename T, uint BlockSize>
kernel void
add2(
    __add2_parameters<T> params,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor3<const T> in1(params.input1_layout, params.input1);
    tensor2<const T> in2(params.input2_layout, params.input2);
    tensor3<T> out(params.output_layout, params.output);

    const uint dim0_size = in2.size(0);
    const uint dim1_size = in2.size(1);
    const uint i = gid.x;
    const uint j = tid.x + gid.y * threadgroup_size.x;

    const uint begin_z = tid.z * BlockSize;
    const uint end_z = begin_z + BlockSize;

    if (j < dim0_size) {
#pragma unroll(BlockSize)
        for (uint k = begin_z; k < end_z && k < dim1_size; k++) {
            out.at(i, j, k) = in1.at(i, j, k) + in2.at(j, k);
        }
    }
}


__lib_metalchat_kernel3_tiled(add2, 8, bfloat);
__lib_metalchat_kernel3_tiled(add2, 16, bfloat);
__lib_metalchat_kernel3_tiled(add2, 32, bfloat);

__lib_metalchat_kernel3_tiled(add2, 8, float);
__lib_metalchat_kernel3_tiled(add2, 16, float);
__lib_metalchat_kernel3_tiled(add2, 32, float);


template <typename T> struct __sub_parameters {
    constant layout2& output_layout;
    device T* output;
    constant layout2& input1_layout;
    device const T* input1;
    constant layout2& input2_layout;
    device const T* input2;
};


template <typename T>
kernel void
sub(__sub_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    tensor2<const T> in1(params.input1_layout, params.input1);
    tensor2<const T> in2(params.input2_layout, params.input2);
    tensor2<T> out(params.output_layout, params.output);

    const uint dim_size = in1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (k < dim_size) {
        out.at(i, k) = in1.at(i, k) - in2.at(i, k);
    }
}


__lib_metalchat_kernel2(sub, bfloat);
__lib_metalchat_kernel2(sub, float);


template <typename T> struct __div_parameters {
    constant layout2& output_layout;
    device T* output;
    constant layout2& input1_layout;
    device const T* input1;
    constant layout2& input2_layout;
    device const T* input2;
};


template <typename T>
kernel void
div(__div_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    tensor2<const T> in1(params.input1_layout, params.input1);
    tensor2<const T> in2(params.input2_layout, params.input2);
    tensor2<T> out(params.output_layout, params.output);

    const uint dim_size = in1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (k < dim_size) {
        out.at(i, k) = in1.at(i, k) / in2.at(i, k);
    }
}


__lib_metalchat_kernel2(div, bfloat);
__lib_metalchat_kernel2(div, float);

// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __hadamard_parameters {
    constant layout2& output_layout;
    device T* output_data;
    constant layout2& input1_layout;
    const device T* input1_data;
    constant layout2& input2_layout;
    const device T* input2_data;
};


template <typename T>
kernel void
hadamard(
    __hadamard_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor2<T> output(params.output_layout, params.output_data);
    tensor2<const T> input1(params.input1_layout, params.input1_data);
    tensor2<const T> input2(params.input2_layout, params.input2_data);

    const uint row_size = input1.size(0);
    const uint dim_size = input1.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input1.at(i, k) * input2.at(i, k);
    }
}


__lib_metalchat_kernel2(hadamard, bfloat);
__lib_metalchat_kernel2(hadamard, float);


template <typename Output, typename Input1, typename Input2>
struct __hadamard_broadcast_parameters {
    tensor2<Output> output;
    tensor2<const Input1> input1;
    tensor2<const Input2> input2;
};


template <typename Output, typename Input1, typename Input2, uint BlockSize>
kernel void
hadamard_broadcast(
    __hadamard_broadcast_parameters<Output, Input1, Input2> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint row_size = params.output.size(0);
    const uint dim_size = params.output.size(1);

    const uint ibegin = (gid.x * threadgroup_size.x + tid.x) * BlockSize;
    const uint iend = ibegin + BlockSize;

    for (uint i = ibegin; i < iend && i < row_size; i++) {
        const uint kbegin = tid.y * BlockSize;
        const uint kend = kbegin + BlockSize;

#pragma unroll(BlockSize)
        for (uint k = kbegin; k < kend && k < dim_size; k++) {
            params.output.at(i, k) =
                (static_cast<Output>(params.input1.at(i, k)) *
                 static_cast<Output>(params.input2.at(i, 0)));
        }
    }
}


__lib_metalchat_kernel2_mixed3_tiled(hadamard_broadcast, 16, bfloat, int8_t, bfloat);
__lib_metalchat_kernel2_mixed3_tiled(hadamard_broadcast, 16, bfloat, int8_t, float);
__lib_metalchat_kernel2_mixed3_tiled(hadamard_broadcast, 16, float, int8_t, bfloat);
__lib_metalchat_kernel2_mixed3_tiled(hadamard_broadcast, 16, float, int8_t, float);


template <typename T> struct __scalar_mul_parameters {
    constant layout2& output_layout;
    device T* output;
    constant layout2& input_layout;
    device const T* input;
    constant const T& multiplier;
};


template <typename T>
kernel void
scalar_mul(
    __scalar_mul_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor2<const T> in(params.input_layout, params.input);
    tensor2<T> out(params.output_layout, params.output);

    const uint row_size = in.size(0);
    const uint dim_size = in.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        out.at(i, k) = in.at(i, k) * params.multiplier;
    }
}


__lib_metalchat_kernel2(scalar_mul, bfloat);
__lib_metalchat_kernel2(scalar_mul, float);

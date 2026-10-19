// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __hadamard_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input1);
    StructuredBuffer(T, 2, input2);
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
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

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
    RWStructuredBuffer(Output, 2, output);
    StructuredBuffer(Input1, 2, input1);
    StructuredBuffer(Input2, 1, input2);
};


template <typename Output, typename Input1, typename Input2>
kernel void
hadamard_broadcast(
    __hadamard_broadcast_parameters<Output, Input1, Input2> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    Tensor(params, output);
    Tensor(params, input1);
    Tensor(params, input2);

    const uint dim0_size = input1.size(0);
    const uint dim1_size = input1.size(1);

    const uint j = gid.x * threadgroup_size.x + tid.x;
    const uint i = gid.y * threadgroup_size.y + tid.y;

    if (i < dim0_size && j < dim1_size) {
        output.at(i, j) =
            (static_cast<Output>(input1.at(i, j)) *
             static_cast<Output>(input2.at(i % input2.size(0))));
    }
}


__lib_metalchat_kernel2_mixed3(hadamard_broadcast, bfloat, int8_t, bfloat);
__lib_metalchat_kernel2_mixed3(hadamard_broadcast, bfloat, int8_t, float);
__lib_metalchat_kernel2_mixed3(hadamard_broadcast, bfloat, bfloat, bfloat);
__lib_metalchat_kernel2_mixed3(hadamard_broadcast, float, int8_t, bfloat);
__lib_metalchat_kernel2_mixed3(hadamard_broadcast, float, int8_t, float);
__lib_metalchat_kernel2_mixed3(hadamard_broadcast, float, float, float);


template <typename T> struct __scalar_mul_parameters {
    RWStructuredBuffer(T, 2, output);
    StructuredBuffer(T, 2, input);
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
    Tensor(params, output);
    Tensor(params, input);

    const uint row_size = input.size(0);
    const uint dim_size = input.size(1);
    const uint i = gid.y * threadgroup_size.y + tid.y;
    const uint k = gid.x * threadgroup_size.x + tid.x;

    if (i < row_size && k < dim_size) {
        output.at(i, k) = input.at(i, k) * params.multiplier;
    }
}


__lib_metalchat_kernel2(scalar_mul, bfloat);
__lib_metalchat_kernel2(scalar_mul, float);

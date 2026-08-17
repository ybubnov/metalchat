// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __conv1d_parameters {
    tensor3<T> output;
    constant layout3& input_layout;
    device const T* input;
    constant layout3& weight_layout;
    device const T* weight;
    constant int& padding;
    constant uint& groups;
};


template <typename T>
kernel void
conv1d(
    __conv1d_parameters<T> params,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]]
)
{
    tensor3<const T> input(params.input_layout, params.input);
    tensor3<const T> weight(params.weight_layout, params.weight);

    const int input_size = static_cast<int>(input.size(2));
    const uint in_channels = weight.size(1) * params.groups;
    const uint in_channels_per_group = weight.size(1);
    const uint out_channels = weight.size(0);
    const uint out_channels_per_group = out_channels / params.groups;
    const uint kernel_size = weight.size(2);

    const uint i = gid.z;
    const uint input_id = gid.x * threadgroup_size.x + tid.x;
    const uint out_channel_id = gid.y * threadgroup_size.y + tid.y;

    const uint output_size = params.output.size(2);

    if (input_id < output_size && out_channel_id < out_channels) {
        T group_conv = T(0);
        uint group_id = out_channel_id / out_channels_per_group;

        for (uint in_channel_id = 0; in_channel_id < in_channels_per_group; in_channel_id++) {
            for (uint k = 0; k < kernel_size; k++) {
                const auto weight_value = weight.at(out_channel_id, in_channel_id, k);

                const int w = static_cast<int>(input_id) - params.padding + k;
                const auto input_value =
                    (w >= 0 && w < input_size)
                        ? input.at(i, group_id * in_channels_per_group + in_channel_id, w)
                        : static_cast<T>(0);

                group_conv += input_value * weight_value;
            }
        }

        params.output.at(i, out_channel_id, input_id) = group_conv;
    }
}


__lib_metalchat_kernel3(conv1d, bfloat);
__lib_metalchat_kernel3(conv1d, float);

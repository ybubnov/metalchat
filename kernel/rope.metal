// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "tensor.h"


#define __rope_parameters(T)                                    \
    constant tensor_layout<2>& output_layout    [[buffer(0)]],  \
    device T* output                            [[buffer(1)]], \
    constant tensor_layout<2>& input_layout     [[buffer(2)]],  \
    device const T* input                       [[buffer(3)]],  \
    constant tensor_layout<2>& freqs_cos_layout [[buffer(4)]],  \
    device const float* freqs_cos               [[buffer(5)]],  \
    constant tensor_layout<2>& freqs_sin_layout [[buffer(6)]],  \
    device const float* freqs_sin               [[buffer(7)]],  \
    constant uint& batch_size                   [[buffer(8)]],  \
    constant uint& n_head                       [[buffer(9)]],  \
    constant uint& start_pos                    [[buffer(10)]], \
    uint gid [[threadgroup_position_in_grid]],                  \
    uint tid [[thread_position_in_threadgroup]]


template <typename T, uint BlockSize>
kernel void
rope(__rope_parameters(T))
{
    tensor<const float, 2> f_cos{freqs_cos, freqs_cos_layout};
    tensor<const float, 2> f_sin{freqs_sin, freqs_sin_layout};
    tensor<const T, 2> in{input, input_layout};

    tensor<T, 2> out{output, output_layout};

    const uint head_dim = f_cos.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    // numel = bs * seq_len * n_head * head_dim
    const uint pos = i / (batch_size * n_head);

    for (uint k = begin; k < end && k < head_dim; k++) {
        float x1 = in.at(i, 2 * k);
        float x2 = in.at(i, 2 * k + 1);

        float fcos = f_cos.at(start_pos + pos, k);
        float fsin = f_sin.at(start_pos + pos, k);

        out.at(i, 2 * k) = T(fcos * x1 - fsin * x2);
        out.at(i, 2 * k + 1) = T(fsin * x1 + fcos * x2);
    }
}


template [[host_name("rope_1_bfloat")]]
kernel void rope<bfloat, 1>(__rope_parameters(bfloat));

template [[host_name("rope_2_bfloat")]]
kernel void rope<bfloat, 2>(__rope_parameters(bfloat));

template [[host_name("rope_4_bfloat")]]
kernel void rope<bfloat, 4>(__rope_parameters(bfloat));

template [[host_name("rope_8_bfloat")]]
kernel void rope<bfloat, 8>(__rope_parameters(bfloat));

template [[host_name("rope_16_bfloat")]]
kernel void rope<bfloat, 16>(__rope_parameters(bfloat));

template [[host_name("rope_32_bfloat")]]
kernel void rope<bfloat, 32>(__rope_parameters(bfloat));


template [[host_name("rope_1_float")]]
kernel void rope<float, 1>(__rope_parameters(float));

template [[host_name("rope_2_float")]]
kernel void rope<float, 2>(__rope_parameters(float));

template [[host_name("rope_4_float")]]
kernel void rope<float, 4>(__rope_parameters(float));

template [[host_name("rope_8_float")]]
kernel void rope<float, 8>(__rope_parameters(float));

template [[host_name("rope_16_float")]]
kernel void rope<float, 16>(__rope_parameters(float));

template [[host_name("rope_32_float")]]
kernel void rope<float, 32>(__rope_parameters(float));

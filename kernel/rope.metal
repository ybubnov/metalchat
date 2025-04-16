// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __rope_parameters {
    constant tensor_layout<2>& output_layout [[buffer(0)]];
    device T* output [[buffer(1)]];
    constant tensor_layout<2>& input_layout [[buffer(2)]];
    device const T* input [[buffer(3)]];
    constant tensor_layout<2>& freqs_cos_layout [[buffer(4)]];
    device const float* freqs_cos [[buffer(5)]];
    constant tensor_layout<2>& freqs_sin_layout [[buffer(6)]];
    device const float* freqs_sin [[buffer(7)]];
    constant uint& batch_size [[buffer(8)]];
    constant uint& n_head [[buffer(9)]];
    constant uint& start_pos [[buffer(10)]];
};


template <typename T, uint BlockSize>
kernel void
rope(
    __rope_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
)
{
    tensor<const float, 2> f_cos{params.freqs_cos, params.freqs_cos_layout};
    tensor<const float, 2> f_sin{params.freqs_sin, params.freqs_sin_layout};
    tensor<const T, 2> in{params.input, params.input_layout};

    tensor<T, 2> out{params.output, params.output_layout};

    const uint head_dim = f_cos.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    // numel = bs * seq_len * n_head * head_dim
    const uint pos = i / (params.batch_size * params.n_head);

    for (uint k = begin; k < end && k < head_dim; k++) {
        float x1 = in.at(i, 2 * k);
        float x2 = in.at(i, 2 * k + 1);

        float fcos = f_cos.at(params.start_pos + pos, k);
        float fsin = f_sin.at(params.start_pos + pos, k);

        out.at(i, 2 * k) = T(fcos * x1 - fsin * x2);
        out.at(i, 2 * k + 1) = T(fsin * x1 + fcos * x2);
    }
}


__lib_metalchat_kernel(rope, bfloat, 8);
__lib_metalchat_kernel(rope, bfloat, 16);
__lib_metalchat_kernel(rope, bfloat, 32);

__lib_metalchat_kernel(rope, float, 8);
__lib_metalchat_kernel(rope, float, 16);
__lib_metalchat_kernel(rope, float, 32);

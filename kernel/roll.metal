// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __roll_parameters {
    tensor1<T> output;
    tensor1<const T> input;
    constant uint& shift;
    constant uint& size;
    constant uint& stride;
};


template <typename T, uint BlockSize>
kernel void
roll(
    __roll_parameters<T> params,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint k = gid * threadgroup_size + tid;

    const uint dim_size = params.size;
    const uint stride_size = params.size * params.stride;
    const uint size = (k / stride_size) * stride_size;

    uint i = (k / params.stride + params.shift) % dim_size;
    uint j = k % params.stride;
    uint m = size + i * params.stride + j;

    if (k < params.input.size(0)) {
        params.output.at(k) = params.input.at(m);
    }
}


__lib_metalchat_kernel(roll, bfloat, 8);
__lib_metalchat_kernel(roll, bfloat, 16);
__lib_metalchat_kernel(roll, bfloat, 32);
__lib_metalchat_kernel(roll, bfloat, 128);

__lib_metalchat_kernel(roll, float, 8);
__lib_metalchat_kernel(roll, float, 16);
__lib_metalchat_kernel(roll, float, 32);
__lib_metalchat_kernel(roll, float, 128);

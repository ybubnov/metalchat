// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T> struct __copy_parameters {
    tensor2<T> output;
    tensor2<const T> input;
};


template <typename T, uint BlockSize>
kernel void
copy(
    __copy_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        params.output.at(i, k) = params.input.at(i, k);
    }
}


__lib_metalchat_kernel2(copy, bfloat, 8);
__lib_metalchat_kernel2(copy, bfloat, 16);
__lib_metalchat_kernel2(copy, bfloat, 32);
__lib_metalchat_kernel2(copy, bfloat, 128);

__lib_metalchat_kernel2(copy, float, 8);
__lib_metalchat_kernel2(copy, float, 16);
__lib_metalchat_kernel2(copy, float, 32);


template <typename T> struct __scatter_parameters {
    tensor2<T> output;
    tensor2<const bool> mask;
    constant T& value;
};


template <typename T, uint BlockSize>
kernel void
scatter(
    __scatter_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.output.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        if (params.mask.at(i, k)) {
            params.output.at(i, k) = params.value;
        }
    }
}

__lib_metalchat_kernel2(scatter, bfloat, 8);
__lib_metalchat_kernel2(scatter, bfloat, 16);
__lib_metalchat_kernel2(scatter, bfloat, 32);
__lib_metalchat_kernel2(scatter, bfloat, 128);

__lib_metalchat_kernel2(scatter, float, 8);
__lib_metalchat_kernel2(scatter, float, 16);
__lib_metalchat_kernel2(scatter, float, 32);


#define __gather_parameters(T)                                  \
    constant layout2& output_layout              [[buffer(0)]], \
    device T* output_data                        [[buffer(1)]], \
    constant layout2& input_layout               [[buffer(2)]], \
    device const T* input_data                   [[buffer(3)]], \
    constant layout2& index_layout               [[buffer(4)]], \
    device const int32_t* index_data             [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]],                 \
    uint2 tid [[thread_position_in_threadgroup]],               \
    uint2 threadgroup_size [[threads_per_threadgroup]]


template <typename T, uint BlockSize>
kernel void
gather(__gather_parameters(T))
{
    tensor2<T> output(output_layout, output_data);
    tensor2<const T> input(input_layout, input_data);
    tensor2<const int32_t> index(index_layout, index_data);

    const uint dim_size = index.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    if (k < dim_size) {
        output.at(i, k) = input.at(i, index.at(i, k));
    }
}


template [[host_name("gather_8_float")]]
kernel void gather<float, 8>(__gather_parameters(float));
template [[host_name("gather_16_float")]]
kernel void gather<float, 16>(__gather_parameters(float));
template [[host_name("gather_32_float")]]
kernel void gather<float, 32>(__gather_parameters(float));


template [[host_name("gather_8_int32_t")]]
kernel void gather<int32_t, 8>(__gather_parameters(int32_t));
template [[host_name("gather_16_int32_t")]]
kernel void gather<int32_t, 16>(__gather_parameters(int32_t));
template [[host_name("gather_32_int32_t")]]
kernel void gather<int32_t, 32>(__gather_parameters(int32_t));

// vi: set filetype=cpp :

#include <metal_stdlib>

#include "kernel.h"
#include "tensor.h"


template <typename T>
inline void
__swap(device T& a, device T& b)
{
    T __t = a;
    a = b;
    b = __t;
}


template <typename T> T inline __ceil_div(T a, T b) { return (a + b - 1) / b; }


template <typename T> struct __sort_parameters {
    tensor2<T> values;
    tensor2<int32_t> indices;
    tensor2<const T> input;
};


template <typename T, uint BlockSize>
kernel void
sort(
    __sort_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
)
{
    const uint dim_size = params.input.size(1);
    const uint dim_size_aligned = params.values.size(1);
    const uint batch = gid.x;

    const uint begin = tid.x * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end; k++) {
        if (k < dim_size) {
            params.values.at(batch, k) = params.input.at(batch, k);
        } else {
            params.values.at(batch, k) = T(-INFINITY);
        }
        params.indices.at(batch, k) = k;
    }

    // k is doubled every iteration
    for (uint k = 2; k <= dim_size_aligned; k = k * 2) {
        // j is halved at every iteration, with truncation of fractional parts
        for (uint j = k >> 1; j > 0; j = j >> 1) {
            threadgroup_barrier(metal::mem_flags::mem_device);

#pragma unroll
            for (uint i = begin; i < end; i++) {
                uint ij = i ^ j;

                device T& value_i = params.values.at(batch, i);
                device T& value_ij = params.values.at(batch, ij);

                if (i < ij) {
                    if (((i & k) == 0) && (value_i < value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(params.indices.at(batch, i), params.indices.at(batch, ij));
                    }
                    if (((i & k) != 0) && (value_i > value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(params.indices.at(batch, i), params.indices.at(batch, ij));
                    }
                }
            }
        }
    }
}


__lib_metalchat_kernel2(sort, bfloat, 8);
__lib_metalchat_kernel2(sort, bfloat, 16);
__lib_metalchat_kernel2(sort, bfloat, 32);
__lib_metalchat_kernel2(sort, bfloat, 128);
__lib_metalchat_kernel2(sort, bfloat, 256);
__lib_metalchat_kernel2(sort, bfloat, 512);
__lib_metalchat_kernel2(sort, bfloat, 1024);
__lib_metalchat_kernel2(sort, bfloat, 2048);


__lib_metalchat_kernel2(sort, float, 8);
__lib_metalchat_kernel2(sort, float, 16);
__lib_metalchat_kernel2(sort, float, 32);
__lib_metalchat_kernel2(sort, float, 512);
__lib_metalchat_kernel2(sort, float, 1024);
__lib_metalchat_kernel2(sort, float, 2048);

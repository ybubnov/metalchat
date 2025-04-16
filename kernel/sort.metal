// vi: set filetype=cpp :

#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __sort_parameters(T)                                        \
    constant tensor_layout<2>& values_layout         [[buffer(0)]], \
    device T* values_ptr                             [[buffer(1)]], \
    constant tensor_layout<2>& indices_layout        [[buffer(2)]], \
    device int32_t* indices_ptr                      [[buffer(3)]], \
    constant tensor_layout<2>& input_layout          [[buffer(4)]], \
    device const T* input_ptr                        [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]],                     \
    uint2 tid [[thread_position_in_threadgroup]]


template <typename T>
inline void
__swap(device T& a, device T& b)
{
    T __t = a;
    a = b;
    b = __t;
}


template <typename T> T inline __ceil_div(T a, T b) { return (a + b - 1) / b; }


template <typename T, uint BlockSize>
kernel void
sort(__sort_parameters(T))
{
    using I = int32_t;

    tensor<const T, 2> in{input_ptr, input_layout};
    tensor<T, 2> values{values_ptr, values_layout};
    tensor<I, 2> indices{indices_ptr, indices_layout};

    const uint dim_size = in.size(1);
    const uint dim_size_aligned = values.size(1);
    const uint batch = gid.x;

    const uint begin = tid.x * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = begin; k < end; k++) {
        if (k < dim_size) {
            values.at(batch, k) = in.at(batch, k);
        } else {
            values.at(batch, k) = T(-INFINITY);
        }
        indices.at(batch, k) = k;
    }

    // k is doubled every iteration
    for (uint k = 2; k <= dim_size_aligned; k = k * 2) {
        // j is halved at every iteration, with truncation of fractional parts
        for (uint j = k >> 1; j > 0; j = j >> 1) {
            threadgroup_barrier(mem_flags::mem_device);

            for (uint i = begin; i < end; i++) {
                uint ij = i ^ j;

                device T& value_i = values.at(batch, i);
                device T& value_ij = values.at(batch, ij);

                if (i < ij) {
                    if (((i & k) == 0) && (value_i < value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(indices.at(batch, i), indices.at(batch, ij));
                    }
                    if (((i & k) != 0) && (value_i > value_ij)) {
                        __swap(value_i, value_ij);
                        __swap(indices.at(batch, i), indices.at(batch, ij));
                    }
                }
            }
        }
    }
}


template [[host_name("sort_1_bfloat")]]
kernel void sort<bfloat, 1>(__sort_parameters(bfloat));

template [[host_name("sort_2_bfloat")]]
kernel void sort<bfloat, 2>(__sort_parameters(bfloat));

template [[host_name("sort_4_bfloat")]]
kernel void sort<bfloat, 4>(__sort_parameters(bfloat));

template [[host_name("sort_8_bfloat")]]
kernel void sort<bfloat, 8>(__sort_parameters(bfloat));

template [[host_name("sort_16_bfloat")]]
kernel void sort<bfloat, 16>(__sort_parameters(bfloat));

template [[host_name("sort_32_bfloat")]]
kernel void sort<bfloat, 32>(__sort_parameters(bfloat));

template [[host_name("sort_512_bfloat")]]
kernel void sort<bfloat, 512>(__sort_parameters(bfloat));

template [[host_name("sort_1024_bfloat")]]
kernel void sort<bfloat, 1024>(__sort_parameters(bfloat));

template [[host_name("sort_2048_bfloat")]]
kernel void sort<bfloat, 2048>(__sort_parameters(bfloat));


template [[host_name("sort_1_float")]]
kernel void sort<float, 1>(__sort_parameters(float));

template [[host_name("sort_2_float")]]
kernel void sort<float, 2>(__sort_parameters(float));

template [[host_name("sort_4_float")]]
kernel void sort<float, 4>(__sort_parameters(float));

template [[host_name("sort_8_float")]]
kernel void sort<float, 8>(__sort_parameters(float));

template [[host_name("sort_16_float")]]
kernel void sort<float, 16>(__sort_parameters(float));

template [[host_name("sort_32_float")]]
kernel void sort<float, 32>(__sort_parameters(float));

template [[host_name("sort_512_float")]]
kernel void sort<float, 512>(__sort_parameters(float));

template [[host_name("sort_1024_float")]]
kernel void sort<float, 1024>(__sort_parameters(float));

template [[host_name("sort_2048_float")]]
kernel void sort<float, 2048>(__sort_parameters(float));

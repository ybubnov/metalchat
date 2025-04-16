// vi: set filetype=cpp :

#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __sort_parameters(T)                                \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device T* output                         [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    constant tensor_layout<2>& buffer_layout [[buffer(4)]], \
    device T* buffer                         [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_index_in_threadgroup]],               \
    uint threadgroup_size [[threads_per_threadgroup]]


#define __swap(a, b) { T __t = a; a = b; b = __t; }


template <typename T> T inline __ceil_div(T a, T b) { return (a + b - 1) / b; }


template <typename T>
inline void
bitonic_sort(thread T* data, uint size)
{
    // k is doubled every iteration
    for (uint k = 2; k <= size; k *= 2) {
        // j is halved at every iteration, with truncation of fractional parts
        for (uint j = k / 2; j > 0; j /= 2) {
            for (uint i = 0; i < size; i++) {
                uint ij = i ^ j;

                if (ij > i) {
                    if ((i & k) == 0 && (data[i] < data[ij])) {
                        __swap(data[i], data[ij]);
                    }
                    if ((i & k) != 0 && (data[i] > data[ij])) {
                        __swap(data[i], data[ij]);
                    }
                }
            }
        }
    }
}


template <typename T>
inline void
merge_sort(thread tensor<T, 2>& src, thread tensor<T, 2>& buf, uint batch, uint begin, uint size)
{
    const uint dim_size = src.size(1);

    uint j = begin;
    uint i0 = begin;
    uint i0_end = begin + size;
    uint i1 = begin + size;
    uint i1_end = min(i1 + size, dim_size);

    while (i0 < i0_end && i1 < i1_end) {
        if (src.at(batch, i0) > src.at(batch, i1)) {
            buf.at(batch, j++) = src.at(batch, i0++);
        } else {
            buf.at(batch, j++) = src.at(batch, i1++);
        }
    }

    while (i0 < i0_end) {
        buf.at(batch, j++) = src.at(batch, i0++);
    }
    while (i1 < i1_end) {
        buf.at(batch, j++) = src.at(batch, i1++);
    }

    uint i_end = min(begin + size * 2, dim_size);
    for (uint i = begin; i < i_end; i++) {
        src.at(batch, i) = buf.at(batch, i);
    }
}


template <typename T, uint BlockSize>
kernel void
sort(__sort_parameters(T))
{
    tensor<const T, 2> in{input, input_layout};
    tensor<T, 2> out{output, output_layout};
    tensor<T, 2> buf{buffer, buffer_layout};

    const uint dim_size = in.size(1);
    const uint batch = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    T local_block[BlockSize];

    for (uint k = begin; k < end && k < dim_size; k++) {
        local_block[k - begin] = in.at(batch, k);
    }

    const uint size = end > dim_size ? dim_size % BlockSize : BlockSize;
    bitonic_sort(local_block, size);

    for (uint k = begin; k < end && k < dim_size; k++) {
        out.at(batch, k) = local_block[k - begin];
    }

    uint block_size = BlockSize;

    for (uint merge_size = __ceil_div(threadgroup_size, uint(2)); merge_size > 0;
         merge_size = __ceil_div(merge_size, uint(2))) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < merge_size) {
            uint i0 = tid * (block_size << 1);
            merge_sort(out, buf, batch, i0, block_size);
        }

        block_size <<= 1;
        if (merge_size == 1) {
            break;
        }
    }
}


template [[host_name("sort_1_bf16")]]
kernel void sort<bfloat, 1>(__sort_parameters(bfloat));

template [[host_name("sort_2_bf16")]]
kernel void sort<bfloat, 2>(__sort_parameters(bfloat));

template [[host_name("sort_4_bf16")]]
kernel void sort<bfloat, 4>(__sort_parameters(bfloat));

template [[host_name("sort_8_bf16")]]
kernel void sort<bfloat, 8>(__sort_parameters(bfloat));

template [[host_name("sort_16_bf16")]]
kernel void sort<bfloat, 16>(__sort_parameters(bfloat));

template [[host_name("sort_32_bf16")]]
kernel void sort<bfloat, 32>(__sort_parameters(bfloat));

template [[host_name("sort_512_bf16")]]
kernel void sort<bfloat, 512>(__sort_parameters(bfloat));

template [[host_name("sort_1024_bf16")]]
kernel void sort<bfloat, 1024>(__sort_parameters(bfloat));

template [[host_name("sort_2048_bf16")]]
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

// vi: set filetype=cpp :

#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __sort_parameters(T)                                        \
    constant tensor_layout<2>& values_layout         [[buffer(0)]], \
    device T* values_ptr                             [[buffer(1)]], \
    constant tensor_layout<2>& values_buffer_layout  [[buffer(2)]], \
    device T* values_buffer_ptr                      [[buffer(3)]], \
    constant tensor_layout<2>& indices_layout        [[buffer(4)]], \
    device int32_t* indices_ptr                      [[buffer(5)]], \
    constant tensor_layout<2>& indices_buffer_layout [[buffer(6)]], \
    device int32_t* indices_buffer_ptr               [[buffer(7)]], \
    constant tensor_layout<2>& input_layout          [[buffer(8)]], \
    device const T* input_ptr                        [[buffer(9)]], \
    uint gid [[threadgroup_position_in_grid]],                      \
    uint tid [[thread_index_in_threadgroup]],                       \
    uint threadgroup_size [[threads_per_threadgroup]]


#define __swap(a, b) { T __t = a; a = b; b = __t; }


template <typename T> T inline __ceil_div(T a, T b) { return (a + b - 1) / b; }


template <typename T>
inline void
bitonic_sort(thread T* values, thread int32_t* indices, uint size)
{
    // k is doubled every iteration
    for (uint k = 2; k <= size; k *= 2) {
        // j is halved at every iteration, with truncation of fractional parts
        for (uint j = k / 2; j > 0; j /= 2) {
            for (uint i = 0; i < size; i++) {
                uint ij = i ^ j;

                if (ij > i) {
                    if ((i & k) == 0 && (values[i] < values[ij])) {
                        __swap(values[i], values[ij]);
                        __swap(indices[i], indices[ij]);
                    }
                    if ((i & k) != 0 && (values[i] > values[ij])) {
                        __swap(values[i], values[ij]);
                        __swap(indices[i], indices[ij]);
                    }
                }
            }
        }
    }
}


template <typename T, uint BlockSize>
inline void
merge_sort(
    thread tensor<T, 2>& values,
    thread tensor<T, 2>& values_buf,
    thread tensor<int32_t, 2>& indices,
    thread tensor<int32_t, 2>& indices_buf,
    uint batch,
    uint begin,
    uint size
)
{
    const uint dim_size = values.size(1);

    uint j = begin;
    uint i0 = begin;
    uint i0_end = begin + size;
    uint i1 = begin + size;
    uint i1_end = min(i1 + size, dim_size);


    while (i0 < i0_end && i1 < i1_end) {
        if (values.at(batch, i0) > values.at(batch, i1)) {
            values_buf.at(batch, j) = values.at(batch, i0);
            indices_buf.at(batch, j++) = indices.at(batch, i0++);
        } else {
            values_buf.at(batch, j) = values.at(batch, i1);
            indices_buf.at(batch, j++) = indices.at(batch, i1++);
        }
    }

    while (i0 < i0_end) {
        values_buf.at(batch, j) = values.at(batch, i0);
        indices_buf.at(batch, j++) = indices.at(batch, i0++);
    }
    while (i1 < i1_end) {
        values_buf.at(batch, j) = values.at(batch, i1);
        indices_buf.at(batch, j++) = indices.at(batch, i1++);
    }

    uint i_end = min(begin + size * 2, dim_size);
    for (uint i = begin, j = 0; i < i_end; i++) {
        values.at(batch, i) = values_buf.at(batch, i);
        indices.at(batch, i) = indices_buf.at(batch, i);
    }
}


/// Sorting algorithm implementation.
///
/// This kernel implements sorting algorithm. The routine is like following: the 2-dimensional
/// tensor is processed row-wise. Each row is split into blocks, so that each block is processed
/// by a thread within a threadgroup.
///
/// Each thread sorts elements of the associated block using bitonic sort function. Then all
/// adjacent blocks are merged using auxiliary memory.
///
/// Implementation note: turned out that in-place merging brings a large burden for arrays of
/// large size (128k elements). Such arrays is used in Llama top-p sampling routine, and therefore
/// we optimize this kernel for speed rather than memory.
template <typename T, uint BlockSize>
kernel void
sort(__sort_parameters(T))
{
    using I = int32_t;

    tensor<const T, 2> in{input_ptr, input_layout};
    tensor<T, 2> values{values_ptr, values_layout};
    tensor<T, 2> values_buf{values_buffer_ptr, values_buffer_layout};
    tensor<I, 2> indices{indices_ptr, indices_layout};
    tensor<I, 2> indices_buf{indices_buffer_ptr, indices_buffer_layout};

    const uint dim_size = in.size(1);
    const uint batch = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    T local_values[BlockSize];
    I local_indices[BlockSize];

    for (uint k = begin; k < end && k < dim_size; k++) {
        local_values[k - begin] = in.at(batch, k);
        local_indices[k - begin] = k;
    }

    const uint size = end > dim_size ? dim_size % BlockSize : BlockSize;
    bitonic_sort(local_values, local_indices, size);

    for (uint k = begin; k < end && k < dim_size; k++) {
        values.at(batch, k) = local_values[k - begin];
        indices.at(batch, k) = local_indices[k - begin];
    }

    uint block_size = BlockSize;

    for (uint merge_size = __ceil_div(threadgroup_size, uint(2)); merge_size > 0;
         merge_size = __ceil_div(merge_size, uint(2))) {

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < merge_size) {
            uint i0 = tid * (block_size << 1);
            merge_sort<T, BlockSize>(
                values, values_buf, indices, indices_buf, batch, i0, block_size
            );
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

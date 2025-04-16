// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>

#include "tensor.h"


using namespace metal;


#define __bmm_parameters(T)                                 \
    constant tensor_layout<2>& mat1_layout [[buffer(0)]],   \
    constant tensor_layout<2>& mat2_layout [[buffer(1)]],   \
    constant tensor_layout<2>& output_layout [[buffer(2)]], \
    device const T* mat1 [[buffer(3)]],                     \
    device const T* mat2 [[buffer(4)]],                     \
    device T* output [[buffer(5)]],                         \
    uint2 group_id [[threadgroup_position_in_grid]],        \
    uint2 thread_id [[thread_position_in_threadgroup]]


/// Matrix multiplication mat1(MxK) @ mat2(KxN) -> C(MxN)
template <typename T>
kernel void
bmm(__bmm_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> m1{mat1, mat1_layout};
    tensor<const T, 2> m2{mat2, mat2_layout};
    tensor<T, 2> out{output, output_layout};

    const uint M = m1.size(0);
    const uint K = m1.size(1);
    const uint N = m2.size(1);

    const uint i = group_id.x * BLOCK_SIZE + thread_id.x;
    const uint j = group_id.y * BLOCK_SIZE + thread_id.y;

    if (i < M && j < N) {
        float partial = 0.0;
        for (uint k = 0; k < K; k++) {
            partial += float(m1.at(i, k)) * float(m2.at(k, j));
        }
        out.at(i, j) = T(partial);
    }
}


template [[host_name("bmm_bf16")]]
kernel void bmm<bfloat>(__bmm_parameters(bfloat));


template [[host_name("bmm_float")]]
kernel void bmm<float>(__bmm_parameters(float));

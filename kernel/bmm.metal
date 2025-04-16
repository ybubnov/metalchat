// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


using namespace metal;


#define __bmm_parameters(T)                                   \
    constant uint& M [[buffer(0)]],                           \
    constant uint& N [[buffer(1)]],                           \
    constant uint& K [[buffer(2)]],                           \
    device const T* mat1 [[buffer(3)]],                       \
    device const T* mat2 [[buffer(4)]],                       \
    device T* output [[buffer(5)]],                           \
    constant uint& mat1_stride0 [[buffer(6)]],                \
    constant uint& mat1_stride1 [[buffer(7)]],                \
    constant uint& mat2_stride0 [[buffer(8)]],                \
    constant uint& mat2_stride1 [[buffer(9)]],                \
    uint2 group_id [[threadgroup_position_in_grid]],          \
    uint2 thread_id [[thread_position_in_threadgroup]]


template <typename T, uint N> struct tensor {
    device T* data;
    uint strides[N];

    device T&
    at(const uint i, const uint j)
    {
        static_assert(N == 2, "at(i, j) only supports 2d tensors");

        uint ptr_offset = strides[0] * i + strides[1] * j;
        return *(data + ptr_offset);
    }
};


/// Matrix multiplication mat1(MxK) @ mat2(KxN) -> C(MxN)
template <typename T>
kernel void
bmm(__bmm_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    tensor<const T, 2> m1{mat1, {mat1_stride0, mat1_stride1}};
    tensor<const T, 2> m2{mat2, {mat2_stride0, mat2_stride1}};
    tensor<T, 2> out{output, {N, 1}};

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

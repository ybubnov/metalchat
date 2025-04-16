// vi: set filetype=cpp :

#include <metal_common>


using namespace metal;


#define __hadamard_parameters(T)               \
    constant uint& dim_size [[buffer(0)]],     \
    device const T* input1 [[buffer(1)]],      \
    device const T* input2 [[buffer(2)]],      \
    device T* output [[buffer(3)]],            \
    uint gid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
hadamard(__hadamard_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    device const T* in1 = input1 + gid * dim_size;
    device const T* in2 = input2 + gid * dim_size;
    device T* out = output + gid * dim_size;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        out[i + j] = in1[i + j] * in2[i + j];
    }
}


template [[host_name("hadamard_bf16")]]
kernel void hadamard<bfloat>(__hadamard_parameters(bfloat));


template [[host_name("hadamard_float")]]
kernel void hadamard<float>(__hadamard_parameters(float));


kernel void
scalar_mul_bf16(
    constant const uint& N [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant const bfloat& multiplier [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < N) {
        output[gid] = input[gid] * multiplier;
    }
}

// vi: set filetype=cpp :

#include <metal_common>


using namespace metal;


#define __sum_parameters(T)                    \
    constant uint& dim_size [[buffer(0)]],     \
    device const T* input1 [[buffer(1)]],      \
    device const T* input2 [[buffer(2)]],      \
    device T* output [[buffer(3)]],            \
    uint gid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]


template <typename T>
kernel void
sum(__sum_parameters(T))
{
    constexpr uint BLOCK_SIZE = 32;

    device const T* in1 = input1 + gid * dim_size;
    device const T* in2 = input2 + gid * dim_size;
    device T* out = output + gid * dim_size;

    uint i = tid * BLOCK_SIZE;
    uint remainder_size = dim_size % BLOCK_SIZE;
    uint block_size = i + BLOCK_SIZE > dim_size ? remainder_size : BLOCK_SIZE;

    for (uint j = 0; j < block_size; j++) {
        out[i + j] = in1[i + j] + in2[i + j];
    }
}


template [[host_name("sum_bf16")]]
kernel void sum<bfloat>(__sum_parameters(bfloat));


template [[host_name("sum_float")]]
kernel void sum<float>(__sum_parameters(float));


#define __sum2_parameters(T)                     \
    constant uint& dim0_size [[buffer(0)]],      \
    constant uint& dim1_size [[buffer(1)]],      \
    device const T* input1 [[buffer(2)]],        \
    device const T* input2 [[buffer(3)]],        \
    device T* output [[buffer(4)]],              \
    uint2 gid [[threadgroup_position_in_grid]],  \
    uint2 tid [[thread_position_in_threadgroup]]


template <typename T>
kernel void
sum2(__sum2_parameters(T))
{
    constexpr uint BLOCK_SIZE_X = 4;
    constexpr uint BLOCK_SIZE_Y = 4;

    device const T* in1 = input1 + gid.x * dim0_size * dim1_size;
    device T* out = output + gid.x * dim0_size * dim1_size;

    uint x = tid.x * BLOCK_SIZE_X;
    uint remainder_size_x = dim0_size % BLOCK_SIZE_X;
    uint block_size_x = x + BLOCK_SIZE_X > dim0_size ? remainder_size_x : BLOCK_SIZE_X;

    uint y = tid.y * BLOCK_SIZE_Y;
    uint remainder_size_y = dim1_size % BLOCK_SIZE_Y;
    uint block_size_y = y + BLOCK_SIZE_Y > dim1_size ? remainder_size_y : BLOCK_SIZE_Y;

    for (uint j = 0; j < block_size_x; j++) {
        for (uint k = 0; k < block_size_y; k++) {
            // out[(x + j) * dim1_size + y + k] = in1[(x + j) * dim1_size + y + k];
            uint index = (x + j) * dim1_size + y + k;
            out[index] = in1[index] + input2[index];
        }
    }
}


template [[host_name("sum2_bf16")]]
kernel void sum2<bfloat>(__sum2_parameters(bfloat));


template [[host_name("sum2_float")]]
kernel void sum2<float>(__sum2_parameters(float));

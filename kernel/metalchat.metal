// vi: set filetype=cpp :

#include <metal_atomic>
#include <metal_stdlib>


using namespace metal;


kernel void
embedding_f16(
    device const int32_t* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const int64_t* stride [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]
)
{
    output[index.x * stride[0] + index.y] = weight[input[index.x] * stride[0] + index.y];
}


kernel void
sum_f16(
    device const bfloat* input [[buffer(0)]],
    volatile device atomic_float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
)
{
    // TODO: converion to float type does not produce correct sum.
    atomic_fetch_add_explicit(output, (float)input[index], memory_order_relaxed);
}


namespace detail {

/// Approximate inverse square root implementation.
bfloat rsqrt(bfloat number)
{
    union {
        bfloat f;
        uint32_t i;
    } conv = { .f = number };

    conv.i  = 0x5f3759df - (conv.i >> 1);
    conv.f *= bfloat(1.5f) - (number * bfloat(0.5F) * conv.f * conv.f);
    return conv.f;
}

} // namespace detail


kernel void
rmsnorm_f16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* eps [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint threadgroup_size [[threads_per_threadgroup]]
)
{
    bfloat input_squared = input[index] * input[index];

    threadgroup bfloat squared_sum = bfloat(0.0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (index == 0) {
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bfloat var = squared_sum / bfloat(threadgroup_size);
    output[index] = weight[index] * detail::rsqrt(var + eps[0]);
}

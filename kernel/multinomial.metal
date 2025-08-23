// vi: set filetype=cpp :

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


/// This is an adapted c++ implementation of the PCG32 random number based on code by
/// (c) 2014 Melissa E. O'Neill / available at http://www.pcg-random.org.
///
/// Licensed under Apache License 2.0.
class pcg32 {
private:
    uint64_t _m_state;
    uint64_t _m_inc;

    uint32_t
    _m_uniform()
    {
        uint64_t pre_state = _m_state;
        // Advance internal state
        _m_state = pre_state * 6364136223846793005ULL + _m_inc;

        // Calculate output function (XSH RR), uses old state for max ILP.
        uint32_t xorshifted = uint32_t(((pre_state >> 18u) ^ pre_state) >> 27u);
        uint32_t rot = uint32_t(pre_state >> 59u);

        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

public:
    pcg32(uint64_t init_state, uint64_t init_seq)
    : _m_state(0),
      _m_inc((init_seq << 1u) | 1u)
    {
        _m_uniform();
        _m_state += init_state;
        _m_uniform();
    }

    float
    uniform()
    {
        union {
            uint32_t u;
            float f;
        } __random;

        __random.u = (_m_uniform() >> 9 | 0x3f800000u);
        return __random.f - 1.0f;
    }
};


template <typename T>
uint
__binary_search(thread tensor2<const T> data, uint batch, T value)
{
    int low = 0;
    int high = data.size(1);

    while (low < high) {
        uint mid = (low + high) / 2;
        T value_m = data.at(batch, mid);

        if (value_m > value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return metal::max(low, 1) - 1;
}


template <typename T> struct __multinomial_parameters {
    tensor2<int32_t> output;
    tensor2<const T> input;
    constant uint64_t& init_state;
    constant uint64_t& init_seq;
};


/// Draw samples from a multinomial distribution.
///
/// Input of this method should a cumulative distribution function of a multinomial
/// distribution, which means that sum of each row of the input should be a equal to 1.0.
///
/// The kernel expects input probabilities to be in reverse order.
template <typename T, uint BlockSize>
kernel void
multinomial(
    __multinomial_parameters<T> params,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]
)
{
    const uint dim_size = params.output.size(1);
    const uint i = gid.x;

    const uint k = tid.x + gid.y * threadgroup_size.x;

    pcg32 generator(params.init_state + i, params.init_seq + k);

    if (k < dim_size) {
        T random = T(generator.uniform());
        params.output.at(i, k) = __binary_search(params.input, i, random);
    }
}


__lib_metalchat_kernel2(multinomial, bfloat, 8);
__lib_metalchat_kernel2(multinomial, bfloat, 16);
__lib_metalchat_kernel2(multinomial, bfloat, 32);

__lib_metalchat_kernel2(multinomial, float, 8);
__lib_metalchat_kernel2(multinomial, float, 16);
__lib_metalchat_kernel2(multinomial, float, 32);

// vi: set filetype=cpp :

#include <metal_common>

#include "tensor.h"


class pcg32 {
private:
    uint64_t _m_state;
    uint64_t _m_inc;

    uint32_t
    _m_uniform()
    {
        uint64_t pre_state = _m_state;
        _m_state = pre_state * 6364136223846793005ULL + _m_inc;

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
__binary_search(thread tensor<const T, 2> data, uint batch, T value)
{
    int low = 0;
    int high = data.size(1) - 1;

    while (low < high) {
        uint mid = (low + high) / 2;
        T value_m = data.at(batch, mid);

        if (value_m < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // return min(uint(low), data.size(1) - 1);
    return low;
}


#define __multinomial_parameters(T)                         \
    constant tensor_layout<2>& output_layout [[buffer(0)]], \
    device int32_t* output                   [[buffer(1)]], \
    constant tensor_layout<2>& input_layout  [[buffer(2)]], \
    device const T* input                    [[buffer(3)]], \
    constant uint64_t& init_state            [[buffer(4)]], \
    constant uint64_t& init_seq              [[buffer(5)]], \
    uint gid [[threadgroup_position_in_grid]],              \
    uint tid [[thread_position_in_threadgroup]]


/// Draw samples from a multinomial distribution.
///
/// Input of this method should a cumulative distribution function of a multinomial
/// distribution, which means that sum of each row of the input should be a equal to 1.0.
template <typename T, uint BlockSize>
kernel void
multinomial(__multinomial_parameters(T))
{
    tensor<const T, 2> in{input, input_layout};
    tensor<int32_t, 2> out{output, output_layout};

    pcg32 generator(init_state + gid, init_seq + tid);

    const uint dim_size = out.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    int32_t local_samples[BlockSize];
    for (uint k = begin, j = 0; k < end && k < dim_size; k++, j++) {
        T random = T(generator.uniform());
        local_samples[j] = __binary_search(in, i, random);
    }

    for (uint k = begin, j = 0; k < end && k < dim_size; k++, j++) {
        out.at(i, k) = local_samples[j];
    }
}


template [[host_name("multinomial_16_float")]]
kernel void multinomial<float, 16>(__multinomial_parameters(float));

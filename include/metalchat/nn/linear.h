#pragma once

#include <iostream>

#include <metalchat/kernel/bmm.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer WeightContainer> class linear {
private:
    tensor<T, 2, WeightContainer> _m_weight;
    bmm<T> _m_bmm;

public:
    linear(linear&&) = default;
    linear(const linear&) = delete;

    linear(tensor<T, 2, WeightContainer>&& weight, device& device)
    : _m_weight(std::move(weight.t())),
      _m_bmm(device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 4, InputContainer>& input)
    {
        return _m_bmm(input, _m_weight);
    }

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 3, InputContainer>& input)
    {
        // A(MxK) @ B(KxN) -> C(MxN)
        assert((input.size(2) == _m_weight.size(0)));

        auto output = empty<T>({input.size(0), input.size(1), _m_weight.size(1)});

        for (auto b0 = 0; b0 < input.size(0); b0++) {
            for (auto i = 0; i < input.size(1); i++) {
                for (auto k = 0; k < _m_weight.size(1); k++) {
                    float partial_sum = 0;
                    for (auto j = 0; j < input.size(2); j++) {
                        partial_sum += input[b0, i, j] * _m_weight[j, k];
                    }
                    output[b0, i, k] = T(partial_sum);
                }
            }
        }

        return output;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(" << l._m_weight.size(0) << ", " << l._m_weight.size(1) << ")";
        os << std::endl << l._m_weight << "::: " << l._m_weight.strides() << std::endl;
        return os;
    }
};


} // namespace nn
} // namespace metalchat

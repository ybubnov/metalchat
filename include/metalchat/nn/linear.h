#pragma once

#include <iostream>

#include <metalchat/kernel/sgemm.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer WeightContainer> class linear {
private:
    tensor<T, 2, WeightContainer> _m_weight;
    sgemm<T> _m_sgemm;

public:
    linear(tensor<T, 2, WeightContainer>&& weight, device& device)
    : _m_weight(std::move(weight)),
      _m_sgemm(device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer> requires(N > 1 && N < 5)
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        return _m_sgemm(input, _m_weight.t());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(" << l._m_weight.size(0) << ", " << l._m_weight.size(1) << ")";
        return os;
    }
};


} // namespace nn
} // namespace metalchat

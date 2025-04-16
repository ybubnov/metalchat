#pragma once

#include <iostream>

#include <metalchat/kernel/bmm.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer WeightContainer> class linear {
private:
    shared_tensor<T, 2, WeightContainer> _m_weight;
    bmm<T> _m_bmm;

public:
    linear(linear&&) = default;
    linear(const linear&) = delete;

    linear(tensor<T, 2, WeightContainer>&& weight, device& device)
    : _m_weight(std::move(weight.t())),
      _m_bmm(device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        return _m_bmm(input, *_m_weight);
    }

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(shared_tensor<T, N, InputContainer> input)
    {
        return _m_bmm(input, _m_weight);
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

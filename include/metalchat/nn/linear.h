#pragma once

#include <iostream>

#include <metalchat/functional.h>


namespace metalchat {
namespace nn {


/// Applies an affine linear transformation to the incoming data.
///
/// This module does not support bias adjustment to the input tensor, and only multiplies
/// it (input) by the specified weight tensor. Meaning it effectively works as matrix
/// multiplication operation.
template <typename T, contiguous_container WeightContainer> class linear {
private:
    shared_tensor<T, 2, WeightContainer> _m_weight;
    device& _m_device;

public:
    linear(shared_tensor<T, 2, WeightContainer> weight, device& device)
    : _m_weight(weight.transpose({1, 0})),
      _m_device(device)
    {}

    linear(tensor<T, 2, WeightContainer>&& weight, device& device)
    : linear(shared_tensor(std::move(weight)), device)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return matmul(input, _m_weight, _m_device);
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

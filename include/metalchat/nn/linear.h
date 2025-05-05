#pragma once

#include <iostream>

#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>


namespace metalchat {
namespace nn {


/// Applies an affine linear transformation to the incoming data.
///
/// This module does not support bias adjustment to the input tensor, and only multiplies
/// it (input) by the specified weight tensor. Meaning it effectively works as matrix
/// multiplication operation.
template <typename T, contiguous_container WeightContainer> class linear : public layer {
private:
    shared_tensor<T, 2, WeightContainer> _m_weight;

public:
    linear(shared_tensor<T, 2, WeightContainer> weight, hardware_accelerator accelerator)
    : layer(accelerator),
      _m_weight(weight)
    {
        register_parameter("weight", _m_weight.get());
    }

    linear(tensor<T, 2, WeightContainer>&& weight, hardware_accelerator accelerator)
    : linear(shared_tensor(std::move(weight)), accelerator)
    {}

    linear(hardware_accelerator accelerator)
    : linear(shared_tensor(tensor<T, 2, WeightContainer>()), accelerator)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return matmul(input, _m_weight.transpose({1, 0}), accelerator());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(" << l._m_weight.sizes() << ")";
        return os;
    }
};


template <typename T, contiguous_container WeightContainer>
using shared_linear = shared_layer<linear<T, WeightContainer>>;


} // namespace nn
} // namespace metalchat

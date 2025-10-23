#pragma once

#include <iostream>

#include <metalchat/functional.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


/// Applies an affine linear transformation to the input data.
///
/// This module does not support bias adjustment to the input tensor, and only multiplies
/// it (input) by the specified weight tensor. Meaning it effectively works as matrix
/// multiplication operation.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class linear : public basic_layer {
public:
    using value_type = T;
    using container_type = Container;
    using weight_type = tensor<T, 2, Container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    using layer_type = linear<T, Container>;
    using layer_pointer = shared_layer_ptr<layer_type>;

    linear(weight_pointer weight_ptr, hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_weight(weight_ptr)
    {
        register_parameter("weight", _M_weight);
    }

    linear(weight_type&& weight, hardware_accelerator& accelerator)
    : linear(shared_tensor(std::move(weight)), accelerator)
    {}

    linear(std::size_t in_features, std::size_t out_features, hardware_accelerator& accelerator)
        requires std::same_as<Container, hardware_memory_container<T>>
    : linear(empty<T>({in_features, out_features}, accelerator), accelerator)
    {}

    linear(hardware_accelerator accelerator)
    : linear(shared_tensor(weight_type()), accelerator)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return matmul(input, _M_weight.transpose({1, 0}), accelerator());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(" << l._M_weight.sizes() << ")";
        return os;
    }

private:
    weight_pointer _M_weight;
};


} // namespace nn
} // namespace metalchat

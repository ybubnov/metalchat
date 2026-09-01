// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iostream>

#include <metalchat/functional.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class basic_linear : public basic_layer {
public:
    using value_type = T;
    using container_type = Container;

    using input_type = future_tensor<value_type, 3>;
    using result_type = future_tensor<value_type, 3>;

    using basic_layer::basic_layer;

    virtual result_type
    operator()(input_type input) = 0;

    virtual ~basic_linear() = default;
};


/// Applies an affine linear transformation to the input data.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class linear : public basic_linear<T, Container> {
public:
    using Linear = basic_linear<T, Container>;

    using value_type = T;
    using container_type = Container;
    using bias_type = tensor<T, 1, Container>;
    using bias_pointer = shared_tensor<T, 1, Container>;
    using weight_type = tensor<T, 2, Container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    /// The linear layer constructor.
    ///
    /// \param in_features The size of each input sample.
    /// \param out_features The size of each output sample.
    /// \param bias If set to true, the layer adds an additive bias.
    /// \param accelerator The hardware accelerator.
    linear(
        std::size_t in_features,
        std::size_t out_features,
        bool bias,
        hardware_accelerator& accelerator
    ) requires std::same_as<Container, hardware_memory_container<T>>
    : linear(
          initialize({out_features, in_features}, accelerator.get_allocator()),
          bias ? initialize({out_features}, accelerator.get_allocator()) : bias_pointer(nullptr),
          accelerator
      )
    {}

    linear(std::size_t in_features, std::size_t out_features, hardware_accelerator& accelerator)
    : linear(in_features, out_features, false, accelerator)
    {}

    linear(hardware_accelerator& accelerator)
    : Linear(accelerator),
      _M_weight(shared_tensor(weight_type())),
      _M_bias()
    {
        Linear::register_parameter("weight", _M_weight);
    }

    /// Enable additive bias.
    ///
    /// The method registers a bias parameter, which is added to the output after
    /// the multiplication operation.
    void
    enable_bias()
    {
        Linear::register_parameter("bias", _M_bias);
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return forward(input);
    }

    Linear::result_type
    operator()(Linear::input_type input)
    {
        return forward(input);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(in_features=" << l._M_weight.size(0) << ", ";
        os << "out_features=" << l._M_weight.size(1) << ", ";
        os << "bias=" << std::boolalpha << bool(l._M_bias) << ")";
        return os;
    }

private:
    linear(weight_pointer weight_ptr, bias_pointer bias_ptr, hardware_accelerator& accelerator)
    : Linear(accelerator),
      _M_weight(weight_ptr),
      _M_bias(bias_ptr)
    {
        Linear::register_parameter("weight", _M_weight);
        if (_M_bias) {
            enable_bias();
        }
    }

    template <allocator_t<void> Allocator, std::size_t N>
    auto
    initialize(std::size_t (&&sizes)[N], Allocator alloc) const
    {
        auto typed_alloc = rebind_allocator<T, Allocator>(alloc);
        return shared_tensor(rand<T>(std::move(sizes), typed_alloc));
    }

    template <allocator_t<T> Allocator, std::size_t N>
    auto
    initialize(std::size_t (&&sizes)[N], Allocator alloc) const
    {
        return shared_tensor(rand<T>(std::move(sizes), alloc));
    }

    template <immutable_tensor_t<T> Input>
    auto
    forward(Input input)
    {
        auto output = matmul(input, _M_weight.transpose({1, 0}), Linear::accelerator());
        if (_M_bias) {
            output = add_broadcast(output, _M_bias, Linear::accelerator());
        }
        return output;
    }

    weight_pointer _M_weight;
    bias_pointer _M_bias;
};


} // namespace nn
} // namespace metalchat

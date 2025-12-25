// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iostream>

#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {

/// Applies Root Mean Square Layer Normalization over a mini-batch of inputs.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class rmsnorm : public basic_layer {
public:
    using value_type = T;
    using container_type = Container;
    using weight_type = tensor<T, 1, Container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    rmsnorm(weight_type&& weight, float eps, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _M_weight(std::move(weight)),
      _M_norm(accelerator),
      _M_eps(eps)
    {
        register_parameter("weight", _M_weight);
    }

    rmsnorm(std::size_t normalized_size, float eps, hardware_accelerator accelerator)
    : rmsnorm(empty<T>({normalized_size}, accelerator), eps, accelerator)
    {}

    rmsnorm(float eps, hardware_accelerator accelerator)
    : rmsnorm(weight_type(), eps, accelerator)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_norm(input, _M_weight, _M_eps);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const rmsnorm& n)
    {
        os << "nn::rmsnorm<" << type_traits<T>::name() << ">";
        os << "(" << n._M_weight.size(0) << ")";
        return os;
    }

private:
    weight_pointer _M_weight;
    kernel::rmsnorm<T> _M_norm;
    float _M_eps;
};


} // namespace nn
} // namespace metalchat

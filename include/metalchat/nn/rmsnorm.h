#pragma once

#include <iostream>

#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/layer.h>


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

    using layer_type = rmsnorm<T, Container>;
    using layer_pointer = shared_layer_ptr<layer_type>;

    rmsnorm(weight_type&& weight, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _M_weight(std::move(weight)),
      _M_norm(accelerator)
    {
        register_parameter("weight", _M_weight);
    }

    rmsnorm(std::size_t normalized_size, hardware_accelerator accelerator)
    : rmsnorm(empty<T>({normalized_size}, accelerator), accelerator)
    {}

    rmsnorm(hardware_accelerator accelerator)
    : rmsnorm(weight_type(), accelerator)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, float eps = 1e-5)
    {
        return _M_norm(input, _M_weight, eps);
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
};


} // namespace nn
} // namespace metalchat

#pragma once

#include <iostream>

#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/layer.h>


namespace metalchat {
namespace nn {

/// Applies Root Mean Square Layer Normalization over a mini-batch of inputs.
template <typename T, contiguous_container Container> class rmsnorm : public basic_layer {
private:
    shared_tensor<T, 1, Container> _m_weight;
    kernel::rmsnorm<T> _m_norm;

public:
    rmsnorm(tensor<T, 1, Container>&& weight, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _m_weight(std::move(weight)),
      _m_norm(accelerator)
    {
        register_parameter("weight", _m_weight);
    }

    rmsnorm(hardware_accelerator accelerator)
    : rmsnorm(tensor<T, 1, Container>(), accelerator)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, float eps = 1e-5)
    {
        return _m_norm(input, _m_weight, eps);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const rmsnorm& n)
    {
        os << "nn::rmsnorm<" << type_traits<T>::name() << ">";
        os << "(" << n._m_weight.size(0) << ")";
        return os;
    }
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
using shared_rmsnorm = shared_layer_ptr<rmsnorm<T, Container>>;


} // namespace nn
} // namespace metalchat

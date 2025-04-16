#pragma once

#include <iostream>

#include <metalchat/kernel/rmsnorm.h>


namespace metalchat {
namespace nn {

template <typename T, ContiguousContainer Container> class rmsnorm {
private:
    shared_tensor<T, 1, Container> _m_weight;
    metalchat::rmsnorm<T> _m_norm;

public:
    rmsnorm(rmsnorm&&) = default;
    rmsnorm(const rmsnorm&) = delete;

    rmsnorm(tensor<T, 1, Container>&& weight, device& device)
    : _m_weight(std::move(weight)),
      _m_norm(device)
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


} // namespace nn
} // namespace metalchat

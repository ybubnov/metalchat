#pragma once

#include <metalchat/container.h>
#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {

template <typename T, ContiguousContainer Container> class rmsnorm {
private:
    metalchat::rmsnorm<T> _m_norm;
    const tensor<T, 1, Container>& _m_weight;

public:
    rmsnorm(const tensor<T, 1, Container>& weight)
    : _m_weight(weight)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input, T eps = T(1e-5))
    {
        return _m_norm(input, _m_weight, eps);
    }
};


} // namespace nn
} // namespace metalchat

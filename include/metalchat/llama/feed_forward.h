#pragma once


#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/silu.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


template <typename T, ContiguousContainer Container> class feed_forward {
private:
    nn::linear<T, Container> m_w1;
    nn::linear<T, Container> m_w2;
    nn::linear<T, Container> m_w3;

    mul<T> m_mul;
    silu<T> m_silu;

public:
    feed_forward(
        const tensor<T, 2, Container>& w1,
        const tensor<T, 2, Container>& w2,
        const tensor<T, 2, Container>& w3,
        device& device
    )
    : m_w1(w1, device),
      m_w2(w2, device),
      m_w3(w3, device),
      m_mul(device),
      m_silu(device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 2, InputContainer>& input)
    {
        return m_w2(m_mul(m_silu(m_w1(input)), m_w3(input)));
    }
};


} // namespace llama
} // namespace metalchat

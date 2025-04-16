#pragma once

#include <iostream>

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

    hadamard<T> m_hadamard;
    silu<T> m_silu;

public:
    feed_forward(feed_forward&&) = default;
    feed_forward(const feed_forward&) = delete;

    feed_forward(
        tensor<T, 2, Container>&& w1,
        tensor<T, 2, Container>&& w2,
        tensor<T, 2, Container>&& w3,
        device& device
    )
    : m_w1(std::move(w1), device),
      m_w2(std::move(w2), device),
      m_w3(std::move(w3), device),
      m_hadamard(device),
      m_silu(device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(shared_tensor<T, 3, InputContainer> input)
    {
        auto input2 = m_w3(input);
        auto input1 = m_silu(m_w1(input).get());

        return m_w2(m_hadamard(input1.get(), input2.get()).get());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const feed_forward& ff)
    {
        os << "llama::feed_forward<" << type_traits<T>::name() << ">()";
        return os;
    }
};


} // namespace llama
} // namespace metalchat

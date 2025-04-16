#pragma once

#include <iostream>

#include <metalchat/functional.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


template <typename T, ContiguousContainer Container> class feed_forward {
private:
    nn::linear<T, Container> _m_w1;
    nn::linear<T, Container> _m_w2;
    nn::linear<T, Container> _m_w3;

    device& _m_device;

public:
    feed_forward(feed_forward&&) = default;
    feed_forward(const feed_forward&) = delete;

    feed_forward(
        tensor<T, 2, Container>&& w1,
        tensor<T, 2, Container>&& w2,
        tensor<T, 2, Container>&& w3,
        device& device
    )
    : _m_w1(std::move(w1), device),
      _m_w2(std::move(w2), device),
      _m_w3(std::move(w3), device),
      _m_device(device)
    {}

    template <immutable_tensor3_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input2 = _m_w3(input);
        auto input1 = fn::silu(_m_w1(input), _m_device);

        return _m_w2(fn::hadamard(input1, input2, _m_device));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const feed_forward&)
    {
        os << "llama::feed_forward<" << type_traits<T>::name() << ">()";
        return os;
    }
};


} // namespace llama
} // namespace metalchat

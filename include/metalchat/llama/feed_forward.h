#pragma once

#include <iostream>

#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


template <typename T, contiguous_container Container> class feed_forward : public layer {
private:
    nn::linear<T, Container> _m_w1;
    nn::linear<T, Container> _m_w2;
    nn::linear<T, Container> _m_w3;

    hardware_accelerator& _m_gpu;

public:
    feed_forward(feed_forward&&) = default;
    feed_forward(const feed_forward&) = delete;

    feed_forward(hardware_accelerator& gpu)
    : layer(),
      _m_w1(gpu),
      _m_w2(gpu),
      _m_w3(gpu),
      _m_gpu(gpu)
    {
        register_layer("w1", _m_w1);
        register_layer("w2", _m_w2);
        register_layer("w3", _m_w3);
    }

    template <immutable_tensor3_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input2 = _m_w3(input);
        auto input1 = silu(_m_w1(input), _m_gpu);

        return _m_w2(hadamard(input1, input2, _m_gpu));
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

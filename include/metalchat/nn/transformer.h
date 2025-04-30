#pragma once

#include <optional>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/linear.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


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
        os << "nn::feed_forward<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container> class transformer : public layer {
private:
    nn::attention<T, Container> _m_attention;
    nn::rmsnorm<T, Container> _m_attention_norm;

    feed_forward<T, Container> _m_ff;
    nn::rmsnorm<T, Container> _m_ff_norm;

    hardware_accelerator& _m_gpu;

public:
    transformer(transformer&&) = default;
    transformer(const transformer&) = delete;

    transformer(attention_options& options, hardware_accelerator& gpu)
    : layer(),
      _m_attention(options, gpu),
      _m_attention_norm(gpu),
      _m_ff(gpu),
      _m_ff_norm(gpu),
      _m_gpu(gpu)
    {
        register_layer("attention", _m_attention);
        register_layer("attention_norm", _m_attention_norm);
        register_layer("feed_forward", _m_ff);
        register_layer("ffn_norm", _m_ff_norm);
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, const std::optional<Mask> mask, std::size_t start_pos = 0)
    {
        auto norm = _m_attention_norm(input);
        auto h = add(input, _m_attention(norm, mask, start_pos), _m_gpu);

        auto ff_norm = _m_ff_norm(h);
        return add(h, _m_ff(ff_norm), _m_gpu);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const transformer&)
    {
        os << "nn::transformer<" << type_traits<T>::name() << ">()";
        return os;
    }
};


} // namespace nn
} // namespace metalchat

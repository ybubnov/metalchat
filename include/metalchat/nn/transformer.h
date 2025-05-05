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
    nn::shared_linear<T, Container> _m_w1;
    nn::shared_linear<T, Container> _m_w2;
    nn::shared_linear<T, Container> _m_w3;

public:
    feed_forward(hardware_accelerator gpu)
    : layer(gpu)
    {
        _m_w1 = register_layer("w1", nn::linear<T, Container>(gpu));
        _m_w2 = register_layer("w2", nn::linear<T, Container>(gpu));
        _m_w3 = register_layer("w3", nn::linear<T, Container>(gpu));
    }

    template <immutable_tensor3_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input2 = _m_w3(input);
        auto input1 = silu(_m_w1(input), accelerator());

        return _m_w2(hadamard(input1, input2, accelerator()));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const feed_forward&)
    {
        os << "nn::feed_forward<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container>
using shared_feed_forward = shared_layer<feed_forward<T, Container>>;


template <typename T, contiguous_container Container> class transformer : public layer {
private:
    nn::shared_attention<T, Container> _m_attention;
    nn::shared_rmsnorm<T, Container> _m_attention_norm;

    nn::shared_feed_forward<T, Container> _m_ff;
    nn::shared_rmsnorm<T, Container> _m_ff_norm;

public:
    transformer(attention_options& options, hardware_accelerator gpu)
    : layer(gpu)
    {
        _m_attention = register_layer("attention", nn::attention<T, Container>(options, gpu));
        _m_attention_norm = register_layer("attention_norm", nn::rmsnorm<T, Container>(gpu));
        _m_ff = register_layer("feed_forward", nn::feed_forward<T, Container>(gpu));
        _m_ff_norm = register_layer("ffn_norm", nn::rmsnorm<T, Container>(gpu));
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, const std::optional<Mask> mask, std::size_t start_pos = 0)
    {
        auto norm = _m_attention_norm(input);
        auto h = add(input, _m_attention(norm, mask, start_pos), accelerator());

        auto ff_norm = _m_ff_norm(h);
        return add(h, _m_ff(ff_norm), accelerator());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const transformer&)
    {
        os << "nn::transformer<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container>
using shared_transformer = shared_layer<transformer<T, Container>>;


} // namespace nn
} // namespace metalchat

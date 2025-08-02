#pragma once

#include <optional>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/linear.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container> class feed_forward : public basic_layer {
private:
    nn::shared_linear<T, Container> _M_w1;
    nn::shared_linear<T, Container> _M_w2;
    nn::shared_linear<T, Container> _M_w3;

public:
    feed_forward(hardware_accelerator gpu)
    : basic_layer(gpu)
    {
        _M_w1 = register_layer("w1", nn::linear<T, Container>(gpu));
        _M_w2 = register_layer("w2", nn::linear<T, Container>(gpu));
        _M_w3 = register_layer("w3", nn::linear<T, Container>(gpu));
    }

    template <immutable_tensor3_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input2 = _M_w3(input);
        auto input1 = silu(_M_w1(input), accelerator());

        return _M_w2(hadamard(input1, input2, accelerator()));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const feed_forward&)
    {
        os << "nn::feed_forward<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container>
using shared_feed_forward = shared_layer_ptr<feed_forward<T, Container>>;


template <typename T, contiguous_container Container> class transformer : public basic_layer {
private:
    nn::shared_attention<T, Container> _M_attention;
    nn::shared_rmsnorm<T, Container> _M_attention_norm;

    nn::shared_feed_forward<T, Container> _M_ff;
    nn::shared_rmsnorm<T, Container> _M_ff_norm;

public:
    transformer(attention_options& options, hardware_accelerator gpu)
    : basic_layer(gpu)
    {
        _M_attention = register_layer("attention", nn::attention<T, Container>(options, gpu));
        _M_attention_norm = register_layer("attention_norm", nn::rmsnorm<T, Container>(gpu));
        _M_ff = register_layer("feed_forward", nn::feed_forward<T, Container>(gpu));
        _M_ff_norm = register_layer("ffn_norm", nn::rmsnorm<T, Container>(gpu));
    }

    template <immutable_tensor3_t<T> Input, cache_t<T> Cache>
    auto
    operator()(Input input, Cache& cache, std::size_t start_pos = 0)
    {
        auto norm = _M_attention_norm(input);
        auto h = add(input, _M_attention(norm, cache, start_pos), accelerator());

        auto ff_norm = _M_ff_norm(h);
        return add(h, _M_ff(ff_norm), accelerator());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const transformer&)
    {
        os << "nn::transformer<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container>
using shared_transformer = shared_layer_ptr<transformer<T, Container>>;


} // namespace nn
} // namespace metalchat

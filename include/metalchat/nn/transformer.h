// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <optional>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/functional.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/layer.h>
#include <metalchat/nn/linear.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class feed_forward : public basic_layer {
private:
    using _Linear = nn::linear<T, Container>;

    _Linear::layer_pointer _M_w1;
    _Linear::layer_pointer _M_w2;
    _Linear::layer_pointer _M_w3;

public:
    using value_type = T;
    using container_type = Container;

    using layer_type = feed_forward<T, Container>;
    using layer_pointer = shared_layer_ptr<layer_type>;

    feed_forward(hardware_accelerator accelerator)
    : basic_layer(accelerator)
    {
        _M_w1 = register_layer("w1", _Linear(accelerator));
        _M_w2 = register_layer("w2", _Linear(accelerator));
        _M_w3 = register_layer("w3", _Linear(accelerator));
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


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class transformer : public basic_layer {
private:
    using _RMSNorm = nn::rmsnorm<T, Container>;
    using _Attention = nn::attention<T, Container>;
    using _FeedForward = nn::feed_forward<T, Container>;

    _Attention::layer_pointer _M_attention;
    _RMSNorm::layer_pointer _M_attention_norm;

    _FeedForward::layer_pointer _M_ff;
    _RMSNorm::layer_pointer _M_ff_norm;

public:
    using value_type = T;
    using container_type = Container;

    using layer_type = transformer<T, Container>;
    using layer_pointer = shared_layer_ptr<layer_type>;

    transformer(const attention_options& options, hardware_accelerator accelerator)
    : basic_layer(accelerator)
    {
        _M_attention = register_layer("attention", _Attention(options, accelerator));
        _M_attention_norm = register_layer("attention_norm", _RMSNorm(accelerator));
        _M_ff = register_layer("feed_forward", _FeedForward(accelerator));
        _M_ff_norm = register_layer("ffn_norm", _RMSNorm(accelerator));
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


} // namespace nn
} // namespace metalchat

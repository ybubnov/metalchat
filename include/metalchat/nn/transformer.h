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
    using BasicLinear = nn::basic_linear<T, Container>;
    using Linear = nn::linear<T, Container>;

    // These layers are declared as polymorphic to provide a way to replace them
    // in runtime with, for example, LoRA linear layer implementations.
    polymorphic_layer<BasicLinear> _M_w1;
    polymorphic_layer<BasicLinear> _M_w2;
    polymorphic_layer<BasicLinear> _M_w3;

public:
    using value_type = T;
    using container_type = Container;

    feed_forward(hardware_accelerator& accelerator)
    : basic_layer(accelerator)
    {}

    void
    initialize()
    {
        _M_w1 = register_polymorphic_layer<BasicLinear, Linear>("w1");
        _M_w2 = register_polymorphic_layer<BasicLinear, Linear>("w2");
        _M_w3 = register_polymorphic_layer<BasicLinear, Linear>("w3");
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
    using RMSNorm = nn::rmsnorm<T, Container>;
    using Attention = nn::attention<T, Container>;
    using FeedForward = nn::feed_forward<T, Container>;

    indirect_layer<Attention> _M_attention;
    indirect_layer<RMSNorm> _M_attention_norm;

    indirect_layer<FeedForward> _M_ff;
    indirect_layer<RMSNorm> _M_ff_norm;

public:
    using value_type = T;
    using container_type = Container;

    transformer(const attention_options& options, hardware_accelerator accelerator)
    : basic_layer(accelerator)
    {
        _M_attention = register_layer<Attention>("attention", options);
        _M_attention_norm = register_layer<RMSNorm>("attention_norm", options.norm_eps);
        _M_ff = register_layer<FeedForward>("feed_forward");
        _M_ff_norm = register_layer<RMSNorm>("ffn_norm", options.norm_eps);
    }

    template <immutable_tensor3_t<T> Input, cache_t<T> Cache>
    auto
    operator()(Input input, Cache& cache, std::size_t start_pos = 0)
    {
        auto norm = _M_attention_norm(input);
        auto h = add(input, _M_attention(norm, cache, start_pos), accelerator());

        auto ff_norm = _M_ff_norm(h);
        auto result = add(h, _M_ff(ff_norm), accelerator());
        return result;
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

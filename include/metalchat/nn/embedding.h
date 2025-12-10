// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cmath>
#include <iostream>
#include <numbers>

#include <metalchat/kernel/embedding.h>
#include <metalchat/nn/layer.h>
#include <metalchat/tensor/shared.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class basic_embedding : public basic_layer {
public:
    using value_type = T;
    using container_type = Container;

    using input_type = future_tensor<int32_t, 2>;
    using result_type = future_tensor<value_type, 3>;

    using basic_layer::basic_layer;

    virtual result_type
    operator()(input_type input);

    template <immutable_tensor2_t<int32_t> Input>
    auto
    operator()(Input input)
    {
        auto alloc = accelerator().get_allocator();
        return operator()(future_tensor(move(input, alloc)));
    }

    virtual ~basic_embedding() = default;
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class embedding : public basic_embedding<T, Container> {
    using _Base = basic_embedding<T, Container>;

public:
    using value_type = T;
    using container_type = T;
    using weight_type = tensor<T, 2, Container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    embedding(weight_pointer weight_ptr, hardware_accelerator& accelerator)
    : _Base(accelerator),
      _M_weight(weight_ptr),
      _M_embedding(accelerator)
    {
        _Base::register_parameter("weight", _M_weight);
    }

    embedding(weight_type&& weight, hardware_accelerator& accelerator)
    : embedding(shared_tensor(std::move(weight)), accelerator)
    {}

    embedding(
        std::size_t num_embeddings, std::size_t embedding_dim, hardware_accelerator& accelerator
    )
    : embedding(empty<T>({num_embeddings, embedding_dim}, accelerator), accelerator)
    {}

    embedding(hardware_accelerator& accelerator)
    : embedding(shared_tensor(weight_type()), accelerator)
    {}

    template <immutable_tensor2_t<int32_t> Input>
    auto
    operator()(Input input)
    {
        return _M_embedding(input, _M_weight);
    }

    _Base::result_type
    operator()(_Base::input_type input)
    {
        return _M_embedding(input, _M_weight);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const embedding& e)
    {
        os << "nn::embedding<" << type_traits<T>::name() << ">";
        os << "(" << e._M_weight.sizes() << ")";
        return os;
    }

private:
    weight_pointer _M_weight;
    kernel::embedding<T> _M_embedding;
};


/// This class implements Rotary Positional Embeddings (RoPE).
///
/// In this implementation we cache the frequencies for each position. When user requests an
/// embedding with start position that is not presented in the cache, the module will recompute
/// the cached frequencies for a range `[start_pos, start_pos + max_seq_len)`.
template <typename T> class rope : public basic_layer {
public:
    using value_type = T;
    using container_type = hardware_memory_container<T>;
    using freqs_type = future_tensor<float, 2>;

private:
    std::size_t _M_start_pos;
    // note: in case llama3.2 _M_dim is equal to 64.
    std::size_t _M_dim;
    std::size_t _M_seq_len;
    float _M_theta;

    freqs_type _M_freqs_cos;
    freqs_type _M_freqs_sin;

    kernel::rope<T> _M_rope;
    kernel::rope_freqs<float> _M_rope_freqs;

    auto
    alloc()
    {
        return future_tensor(empty<float>({_M_seq_len, _M_dim / 2}, accelerator().get_allocator()));
    }

    void
    scale_freqs(
        std::vector<float>& freqs,
        float scale = 8.0,
        float low_scale = 1.0,
        float high_scale = 4.0,
        float context_length = 8192.0
    )
    {
        auto low_wavelen = context_length / low_scale;
        auto high_wavelen = context_length / high_scale;

        for (std::size_t i = 0; i < freqs.size(); i++) {
            auto wavelen = 2 * std::numbers::pi / freqs[i];

            if (wavelen < high_wavelen) {
                continue;
            } else if (wavelen > low_wavelen) {
                freqs[i] = freqs[i] / scale;
            } else {
                auto smoothing = (context_length / wavelen - low_scale) / (high_scale - low_scale);
                freqs[i] = (1 - smoothing) * freqs[i] / scale + smoothing * freqs[i];
            }
        }
    }

    void
    update(std::size_t start_pos)
    {
        _M_start_pos = start_pos;
        std::tie(_M_freqs_cos, _M_freqs_sin) = _M_rope_freqs(_M_freqs_cos, _M_freqs_sin, start_pos);
    }

public:
    rope(std::size_t dim, std::size_t max_seq_len, float theta, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _M_start_pos(0),
      _M_dim(dim),
      _M_seq_len(max_seq_len * 2),
      _M_theta(theta),
      _M_freqs_cos(alloc()),
      _M_freqs_sin(alloc()),
      _M_rope(accelerator),
      _M_rope_freqs(dim, _M_seq_len, theta, accelerator)
    {
        update(0);
    }

    template <immutable_tensor4_t<T> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        if (_M_dim != input.sizes().back()) {
            throw std::invalid_argument(std::format(
                "nn::rope: the last dimensions has wrong size {} != {}", _M_dim,
                input.sizes().back()
            ));
        }

        // When the requested start position is outside of the frequencies range, recompute
        // the frequencies for a new position.
        if (start_pos < _M_start_pos || start_pos >= _M_start_pos + _M_seq_len) {
            update(start_pos);
        }

        return _M_rope(input, _M_freqs_cos, _M_freqs_sin, start_pos - _M_start_pos);
    }
};


} // namespace nn
} // namespace metalchat

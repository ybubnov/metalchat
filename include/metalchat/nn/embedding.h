#pragma once

#include <cmath>
#include <iostream>
#include <numbers>

#include <metalchat/kernel/embedding.h>
#include <metalchat/layer.h>
#include <metalchat/tensor/shared.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container> class embedding : public basic_layer {
private:
    shared_tensor<T, 2, Container> _m_weight;
    kernel::embedding<T> _m_embedding;

public:
    embedding(shared_tensor<T, 2, Container> weight, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _m_weight(weight),
      _m_embedding(accelerator)
    {
        register_parameter("weight", _m_weight.get());
    }

    embedding(tensor<T, 2, Container>&& weight, hardware_accelerator accelerator)
    : embedding(shared_tensor(std::move(weight)), accelerator)
    {}

    embedding(hardware_accelerator accelerator)
    : embedding(shared_tensor(tensor<T, 2, Container>()), accelerator)
    {}

    template <immutable_tensor2_t<int32_t> Input>
    auto
    operator()(Input input)
    {
        return _m_embedding(input, _m_weight);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const embedding& e)
    {
        os << "nn::embedding<" << type_traits<T>::name() << ">";
        os << "(" << e._m_weight.sizes() << ")";
        return os;
    }
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
using shared_embedding = shared_layer<embedding<T, Container>>;


/// This class implements Rotary Positional Embeddings (RoPE).
///
/// In this implementation we cache the frequencies for each position. When user requests an
/// embedding with start position that is not presented in the cache, the module will recompute
/// the cached frequencies for a range `[start_pos, start_pos + max_seq_len)`.
template <typename T> class rope : public basic_layer {
public:
    using value_type = T;
    using freqs_tensor = future_tensor<float, 2>;

private:
    std::size_t _m_start_pos;
    // note: in case llama3.2 _m_dim is equal to 64.
    std::size_t _m_dim;
    std::size_t _m_seq_len;
    float _m_theta;

    freqs_tensor _m_freqs_cos;
    freqs_tensor _m_freqs_sin;

    kernel::rope<T> _m_rope;
    kernel::rope_freqs<float> _m_rope_freqs;

    auto
    alloc()
    {
        return future_tensor(empty<float>({_m_seq_len, _m_dim / 2}, accelerator().get_allocator()));
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
        _m_start_pos = start_pos;
        std::tie(_m_freqs_cos, _m_freqs_sin) = _m_rope_freqs(_m_freqs_cos, _m_freqs_sin, start_pos);
    }

public:
    rope(std::size_t dim, std::size_t max_seq_len, float theta, hardware_accelerator accelerator)
    : basic_layer(accelerator),
      _m_start_pos(0),
      _m_dim(dim),
      _m_seq_len(max_seq_len * 2),
      _m_theta(theta),
      _m_freqs_cos(alloc()),
      _m_freqs_sin(alloc()),
      _m_rope(accelerator),
      _m_rope_freqs(dim, _m_seq_len, theta, accelerator)
    {
        update(0);
    }

    template <immutable_tensor4_t<T> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        if (_m_dim != input.sizes().back()) {
            throw std::invalid_argument(std::format(
                "nn::rope: the last dimensions has wrong size {} != {}", _m_dim,
                input.sizes().back()
            ));
        }

        // When the requested start position is outside of the frequencies range, recompute
        // the frequencies for a new position.
        if (start_pos < _m_start_pos || start_pos >= _m_start_pos + _m_seq_len) {
            update(start_pos);
        }

        return _m_rope(input, _m_freqs_cos, _m_freqs_sin, start_pos - _m_start_pos);
    }
};


} // namespace nn
} // namespace metalchat

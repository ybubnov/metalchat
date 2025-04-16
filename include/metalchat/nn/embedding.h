#pragma once

#include <cmath>
#include <iostream>
#include <numbers>

#include <metalchat/functional.h>
#include <metalchat/kernel/embedding.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer Container> class embedding {
private:
    tensor<T, 2, Container> _m_weight;
    metalchat::embedding<T> _m_embedding;

public:
    embedding(embedding&&) = default;

    embedding(tensor<T, 2, Container>&& weight, device& device)
    : _m_weight(std::move(weight)),
      _m_embedding(device)
    {}

    template <integral IndexType, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<IndexType, 2, InputContainer>& input)
    {
        return _m_embedding(input, _m_weight);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const embedding& e)
    {
        os << "nn::embedding<" << type_traits<T>::name() << ">";
        os << "(" << e._m_weight.size(0) << ", " << e._m_weight.size(1) << ")";
        return os;
    }
};


template <typename T> class rope {
private:
    // note: in case llama3.2 _m_dim is equal to 64.
    std::size_t _m_dim;
    std::size_t _m_max_seq_len;
    float _m_theta;

    tensor<float, 2, owning_ref<float>> _m_freqs_cos;
    tensor<float, 2, owning_ref<float>> _m_freqs_sin;

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

        for (auto i = 0; i < freqs.size(); i++) {
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

public:
    rope(rope&&) = default;

    rope(std::size_t dim, std::size_t max_seq_len, float theta)
    : _m_dim(dim),
      _m_max_seq_len(max_seq_len),
      _m_theta(theta),
      _m_freqs_cos(empty<float>({_m_max_seq_len * 2, _m_dim / 2})),
      _m_freqs_sin(empty<float>({_m_max_seq_len * 2, _m_dim / 2}))
    {
        std::vector<float> freqs(_m_dim / 2);
        for (std::size_t i = 0; i < freqs.size(); i++) {
            freqs[i] = 1.0f / std::powf(_m_theta, 2.0 * i / _m_dim);
        }

        // scale_freqs(freqs);

        for (auto i = 0; i < _m_max_seq_len * 2; i++) {
            for (auto j = 0; j < _m_dim / 2; j++) {
                float angle = float(i) * freqs[j];
                _m_freqs_cos[i, j] = std::cos(angle);
                _m_freqs_sin[i, j] = std::sin(angle);
            }
        }
    }

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 4, InputContainer>& input, std::size_t start_pos = 0)
    {
        if (_m_dim != input.sizes().back()) {
            throw std::invalid_argument(std::format(
                "nn::rope: the last dimensions has wrong size {} != {}", _m_dim,
                input.sizes().back()
            ));
        }

        // bs, seq_len, n_head, head_dim
        auto output = empty_like(input);
        for (auto bs = 0; bs < input.size(0); bs++) {
            for (auto i = 0; i < input.size(1); i++) {
                for (auto j = 0; j < input.size(2); j++) {
                    for (auto k = 0; k < _m_dim / 2; k++) {
                        auto x1 = input[bs, i, j, 2 * k];
                        auto x2 = input[bs, i, j, 2 * k + 1];
                        auto cos = _m_freqs_cos[start_pos + i, k];
                        auto sin = _m_freqs_sin[start_pos + i, k];

                        output[bs, i, j, 2 * k] = cos * x1 - sin * x2;
                        output[bs, i, j, 2 * k + 1] = sin * x1 + cos * x2;
                    }
                }
            }
        }
        return output;
    }
};


} // namespace nn
} // namespace metalchat

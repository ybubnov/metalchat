// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cmath>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/activation.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer_array.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>


namespace metalchat {
namespace nn {


struct gemma3_options {
    std::size_t head_dim = 0;
    std::size_t n_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t n_layers = 0;
    std::size_t max_seq_len = 0;
    std::size_t sliding_window = 0;
    std::size_t sliding_stride = 0;
    float attn_scale = 0.0f;
    float rope_theta = 0.0f;
    float rope_sliding_theta = 0.0f;
    float norm_eps = 0.0f;
};


/// Gemma is a family of lightweight open models from Google, built from the same research
/// and technology used to create the Gemini models.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class gemma3 : public basic_layer {
private:
    using Transformer = nn::transformer<T, Container, kernel::gelu<T>>;
    using TransformerArray = nn::layer_array<Transformer>;
    using Embedding = nn::embedding<T, Container>;
    using RMSNorm = nn::rmsnorm<T, Container>;
    using Linear = nn::linear<T, Container>;

    indirect_layer<Embedding> _M_embedding;
    indirect_layer<Linear> _M_output;

    indirect_layer<RMSNorm> _M_norm;
    indirect_layer<TransformerArray> _M_transforms;

    gemma3_options _M_options;

    inline bool
    is_sliding(std::size_t i) const
    {
        return (i + 1) % _M_options.sliding_stride;
    }

public:
    using index_type = int32_t;
    using value_type = T;
    using container_type = Container;
    using tensor_type = future_tensor<index_type, 2>;

    gemma3(const gemma3_options& options, hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_options(options)
    {
        _M_norm = register_layer<RMSNorm>("norm", options.norm_eps);
        _M_transforms = register_layer<TransformerArray>("layers");
        _M_embedding = register_layer<Embedding>("tok_embeddings");
        _M_output = register_layer<Linear>("output");

        for (std::size_t i = 0; i < options.n_layers; i++) {
            auto rope_theta = is_sliding(i) ? options.rope_sliding_theta : options.rope_theta;

            attention_options attention_opts{
                .head_dim = options.head_dim,
                .n_heads = options.n_heads,
                .n_kv_heads = options.n_kv_heads,
                .max_seq_len = options.max_seq_len,
                .max_batch_size = 1,
                .rope_theta = rope_theta,
                .scale = float(1.0 / std::sqrt(double(options.attn_scale))),
                .norm_eps = options.norm_eps,
            };

            _M_transforms->emplace_back(attention_opts, accelerator);
            _M_transforms->back().enable_norm(options.norm_eps);
            _M_transforms->back().enable_post_norm(options.norm_eps);
        }
    }

    template <immutable_hardware_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);
        x = mul(x, T(std::sqrt(640.0f)), accelerator());

        auto len = x.size(1);
        auto end_pos = std::min(start_pos + len, _M_options.max_seq_len);

        auto rolling_mask = make_causal_mask<T>(len, end_pos, accelerator());
        auto sliding_mask = make_sliding_causal_mask<T>(
            len, end_pos, /*window=*/_M_options.sliding_window, accelerator()
        );

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            auto& mask = is_sliding(i) ? sliding_mask : rolling_mask;
            x = transform(x, mask, start_pos);
        }

        auto output = _M_norm(x);

        len = output.size(1);
        output = output.narrow(1, len - 1, 1);

        return _M_output(output);
    }

    template <immutable_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto alloc = accelerator().get_allocator();
        return operator()(future_tensor(move(input, alloc)), start_pos);
    }
};


} // namespace nn
} // namespace metalchat

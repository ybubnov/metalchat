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


struct lfm2_options {
    std::size_t head_dim = 0;
    std::size_t hidden_dim = 0;
    std::size_t n_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t kernel_size = 0;
    std::size_t max_seq_len = 0;
    std::size_t max_batch_size = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
    std::vector<attentionkind> attentions = {};
};


/// LFM2 is a family of hybrid models designed for on-device deployment. It builds on the LFM2
/// architecture with extended pre-training and reinforcement learning.
///
/// For more details refer to the `Liquid AI Documentation <https://www.liquid.ai/models>`_.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class lfm2 : public basic_layer {
    using MultiheadAttention = multihead_attention<T, Container>;
    using RecurrentAttention = recurrent_attention<T, Container>;

    using Transformer = transformer<T, Container, kernel::silu<T>>;
    using TransformerArray = layer_array<Transformer>;
    using Embedding = embedding<T, Container>;
    using RotaryPositionalEmbedding = rope<T>;
    using RMSNorm = rmsnorm<T, Container>;
    using Linear = linear<T, Container>;

    indirect_layer<Embedding> _M_embedding;
    indirect_layer<Linear> _M_output;

    indirect_layer<RMSNorm> _M_norm;
    indirect_layer<TransformerArray> _M_transforms;

    lfm2_options _M_options;

    bool
    uses_recurrent_attention(std::size_t i)
    {
        return _M_options.attentions[i] == attention::recurrent;
    }

public:
    using index_type = int32_t;
    using value_type = T;
    using container_type = Container;
    using tensor_type = future_tensor<index_type, 2>;

    lfm2(const lfm2_options& options, hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_options(options)
    {
        _M_norm = register_layer<RMSNorm>("norm", options.norm_eps, /*mu=*/1.0f);
        _M_transforms = register_layer<TransformerArray>("layers");
        _M_embedding = register_layer<Embedding>("tok_embeddings");
        _M_output = register_layer<Linear>("output");

        // Reuse positional encodings across all network layers.
        indirect_layer<RotaryPositionalEmbedding> rope(
            options.head_dim, options.max_seq_len, options.rope_theta, accelerator
        );

        multihead_attention_options mha_options{
            .head_dim = options.head_dim,
            .n_heads = options.n_heads,
            .n_kv_heads = options.n_kv_heads,
            .max_seq_len = options.max_seq_len,
            .max_batch_size = options.max_batch_size,
            .rope_theta = options.rope_theta,
            .scale = 1.0f / std::sqrt(float(options.head_dim)),
            // Enable key and value RMS normalization in multi-head attention.
            .norm_eps = options.norm_eps,
            .norm_mu = 0.0f
        };

        recurrent_attention_options ra_options{
            .hidden_dim = options.hidden_dim,
            .kernel_size = options.kernel_size,
            .groups = options.hidden_dim,
            .max_seq_len = options.max_seq_len,
            .max_batch_size = options.max_batch_size
        };

        for (std::size_t i = 0; i < options.attentions.size(); i++) {
            if (uses_recurrent_attention(i)) {
                _M_transforms->emplace_back(
                    indirect_layer<RecurrentAttention>(ra_options, accelerator)
                );
            } else {
                _M_transforms->emplace_back(indirect_layer<MultiheadAttention>(mha_options, rope));
            }

            // Enable attention (operator) norm and block normalization (ffn).
            _M_transforms->back().enable_norm(options.norm_eps);
        }
    }

    template <immutable_hardware_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);

        auto len = x.size(1);
        auto end_pos = std::min(start_pos + len, _M_options.max_seq_len);

        auto causal_mask = make_causal_mask<T>(len, end_pos, accelerator());

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            const auto& mask = uses_recurrent_attention(i) ? std::nullopt : causal_mask;
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

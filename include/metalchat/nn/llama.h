// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cmath>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer_array.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>


namespace metalchat {
namespace nn {


/// Configuration options for Llama3 model.
struct llama3_options {
    std::size_t head_dim = 0;
    std::size_t n_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t n_layers = 0;
    std::size_t max_seq_len = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
};


llama3_options
default_llama3_1b_options();


/// Llama 3 is an auto-regressive language model that uses an optimized transformer architecture.
/// The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human
/// feedback (RLHF) to align with human preferences for helpfulness and safety.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class llama3 : public basic_layer {
private:
    using Attention = nn::attention<T, Container>;
    using Transformer = nn::transformer<T, Container>;
    using TransformerArray = nn::layer_array<Transformer>;
    using BasicEmbedding = nn::basic_embedding<T, Container>;
    using Embedding = nn::embedding<T, Container>;
    using RotaryPositionalEmbedding = nn::rope<T>;
    using RMSNorm = nn::rmsnorm<T, Container>;
    using BasicLinear = nn::basic_linear<T, Container>;
    using Linear = nn::linear<T, Container>;

    polymorphic_layer<BasicEmbedding> _M_embedding;
    polymorphic_layer<BasicLinear> _M_output;

    indirect_layer<RMSNorm> _M_norm;
    indirect_layer<TransformerArray> _M_transforms;

    llama3_options _M_options;

public:
    using index_type = int32_t;
    using value_type = T;
    using container_type = Container;
    using tensor_type = future_tensor<index_type, 2>;

    /// Constructs a new Llama3 model with uninitialized weights with the given options.
    llama3(const llama3_options& options, hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_options(options)
    {
        _M_norm = register_layer<RMSNorm>("norm", options.norm_eps);
        _M_transforms = register_layer<TransformerArray>("layers");

        _M_embedding = register_polymorphic_layer<Embedding>("tok_embeddings");
        _M_output = register_polymorphic_layer<Linear>("output");

        attention_options attention_opts{
            .head_dim = options.head_dim,
            .n_heads = options.n_heads,
            .n_kv_heads = options.n_kv_heads,
            .max_seq_len = options.max_seq_len,
            .max_batch_size = 1,
            .rope_theta = options.rope_theta,
            .scale = 1.0f / std::sqrt(float(options.head_dim)),
            // Llama3 models does not implement RMS-normalization of keys
            // and queries in the attention layer, so we disable it here.
            .norm_eps = std::nullopt,
            .norm_mu = std::nullopt,
        };

        indirect_layer<RotaryPositionalEmbedding> rope(
            options.head_dim, options.max_seq_len, options.rope_theta, accelerator
        );

        for (std::size_t i = 0; i < options.n_layers; i++) {
            indirect_layer<Attention> attention(attention_opts, rope);
            _M_transforms->emplace_back(attention);
            _M_transforms->back().enable_norm(options.norm_eps);
        }
    }

    /// Invoke the layer.
    ///
    /// \tparam Input type of the input tensor.
    /// \param input a 2-dimensional tensor with the indices of the input tokens.
    /// \param start_pos a start position of the input sequence.
    ///
    /// \returns a \ref future_tensor with logits of model vocabulary.
    template <immutable_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);

        auto len = x.size(1);
        auto end_pos = std::min(start_pos + len, _M_options.max_seq_len);
        auto mask = make_causal_mask<T>(len, end_pos, accelerator());

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            x = transform(x, mask, start_pos);
        }

        auto output = _M_norm(x);

        len = output.size(1);
        output = output.narrow(1, len - 1, 1);

        return _M_output(output);
    }
};


} // namespace nn
} // namespace metalchat

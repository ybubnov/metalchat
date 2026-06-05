// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cmath>
#include <list>
#include <optional>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer.h>
#include <metalchat/nn/options.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace nn {


/// Llama 3 is an auto-regressive language model that uses an optimized transformer architecture.
/// The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human
/// feedback (RLHF) to align with human preferences for helpfulness and safety.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class llama3 : public basic_layer {
private:
    using Transformer = nn::transformer<T, Container>;
    using TransformerArray = layer_array<Transformer>;
    using BasicEmbedding = nn::basic_embedding<T, Container>;
    using Embedding = nn::embedding<T, Container>;
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
        _M_norm = register_layer<RMSNorm>("norm", options.norm_eps());
        _M_transforms = register_layer<TransformerArray>("layers");

        _M_embedding = register_polymorphic_layer<Embedding>("tok_embeddings");
        _M_output = register_polymorphic_layer<Linear>("output");

        attention_options attention_opts{
            .head_dim = options.head_dim(),
            .n_heads = options.n_heads(),
            .n_kv_heads = options.n_kv_heads(),
            .max_seq_len = options.max_seq_len(),
            .max_batch_size = 1,
            .rope_theta = options.rope_theta(),
            .scale = 1.0f / std::sqrt(float(options.head_dim())),
            // Llama3 models does not implement RMS-normalization of keys
            // and queries in the attention layer, so we disable it here.
            .norm_eps = std::nullopt
        };

        for (std::size_t i = 0; i < options.n_layers(); i++) {
            _M_transforms->emplace_back(attention_opts, accelerator);
            _M_transforms->back().enable_norm(options.norm_eps());
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
        auto end_pos = std::min(start_pos + len, _M_options.max_seq_len());
        auto mask = create_additive_causal_mask(len, end_pos);

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            x = transform(x, mask, start_pos);
        }

        auto output = _M_norm(x);

        len = output.size(1);
        output = output.narrow(1, len - 1, 1);

        return _M_output(output);
    }

    auto
    create_additive_causal_mask(std::size_t len, std::size_t end_pos) const
    {
        std::optional<future_tensor<T, 2>> mask;

        if (len > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());

            auto alloc = accelerator().get_allocator();
            auto m = full<T>({len, end_pos}, -infinity, alloc);

            triu(m.narrow(1, end_pos - len, len), /*diagonal=*/1);
            mask = std::make_optional(std::move(m));
        }

        return mask;
    }
};


} // namespace nn
} // namespace metalchat

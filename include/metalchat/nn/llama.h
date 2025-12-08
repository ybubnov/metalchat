// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <list>
#include <optional>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer.h>
#include <metalchat/nn/options.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/sampling.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace nn {


// The original implementation of Llama 3.2 shares the weight of token embeddings and the output
// layer, use a shared tensor in order to reduce memory footprint.
struct metallama3_document_adaptor {
    void
    adapt(safetensor_document& document) const
    {
        document.insert("output.weight", "tok_embeddings.weight");
    }
};


/// Llama 3 is an auto-regressive language model that uses an optimized transformer architecture.
/// The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human
/// feedback (RLHF) to align with human preferences for helpfulness and safety.
template <
    typename T,
    contiguous_container Container = hardware_memory_container<T>,
    cache_t<T> Cache = sink_cache<T>>
class llama3 : public basic_layer {
private:
    using Transformer = nn::transformer<T, Container>;
    using TransformerArray = layer_array<Transformer>;
    using CacheArray = layer_array<Cache>;
    using BasicEmbedding = nn::basic_embedding<T, Container>;
    using Embedding = nn::embedding<T, Container>;
    using RMSNorm = nn::rmsnorm<T, Container>;
    using BasicLinear = nn::basic_linear<T, Container>;
    using Linear = nn::linear<T, Container>;

    polymorphic_layer<BasicEmbedding> _M_embedding;
    polymorphic_layer<BasicLinear> _M_output;

    indirect_layer<RMSNorm> _M_norm;
    indirect_layer<TransformerArray> _M_transforms;
    indirect_layer<CacheArray> _M_caches;

    std::shared_ptr<basic_sampler<T>> _M_sampler;

public:
    using index_type = int32_t;
    using value_type = T;
    using container_type = Container;
    using cache_type = Cache;
    using tensor_type = future_tensor<index_type, 2>;

    /// Constructs a new Llama3 model with uninitialized weights with the given options.
    llama3(const llama3_options& options, hardware_accelerator& accelerator)
        requires cache_constructible<Cache>
    : basic_layer(accelerator),
      _M_sampler(std::make_shared<nucleus_sampler<value_type>>())
    {
        _M_norm = register_layer<RMSNorm>("norm");
        _M_transforms = register_layer<TransformerArray>("layers");
        _M_caches = register_layer<CacheArray>("caches");

        const auto caching_opts = caching_options{
            .head_dim = options.head_dim(),
            .n_heads = options.n_heads(),
            .n_kv_heads = options.n_kv_heads(),
            .max_seq_len = options.max_seq_len(),
            .max_batch_size = 1
        };

        const auto attention_opts = attention_options{
            .head_dim = options.head_dim(),
            .n_heads = options.n_heads(),
            .n_kv_heads = options.n_kv_heads(),
            .max_seq_len = options.max_seq_len(),
            .rope_theta = options.rope_theta()
        };

        for (std::size_t i = 0; i < options.n_layers(); i++) {
            _M_transforms->emplace_back(attention_opts, accelerator);
            _M_caches->emplace_back(caching_opts, accelerator);
        }
    }

    void
    initialize()
    {
        _M_embedding = register_polymorphic_layer<BasicEmbedding, Embedding>("tok_embeddings");
        _M_output = register_polymorphic_layer<BasicLinear, Linear>("output");
    }

    template <immutable_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);
        std::cout << "embedding=" << x.get() << std::endl;

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            auto& cache = _M_caches->at(i);

            x = transform(x, cache, start_pos);
            std::cout << "transform[" << i << "]=" << x.get() << std::endl;
        }

        auto output = _M_norm(x);

        auto len = output.size(1);
        output = output.narrow(1, len - 1, 1);

        return _M_output(output);
    }

    tensor_type
    transform(tensor_type input, std::size_t start_pos = 0)
    {
        auto logits = operator()(input, start_pos);
        return _M_sampler->sample(logits.template flatten<2>(), accelerator());
    }

    template <safetensor_document_adaptor SafetensorDocumentAdaptor = metallama3_document_adaptor>
    void
    load(
        const std::filesystem::path& path,
        SafetensorDocumentAdaptor document_adaptor = SafetensorDocumentAdaptor()
    )
    {
        auto document = safetensor_document::open(path, accelerator());
        document_adaptor.adapt(document);
        document.load(*this);
    }
};


} // namespace nn
} // namespace metalchat

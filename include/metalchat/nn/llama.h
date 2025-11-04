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
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/sampling.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace nn {


struct llama3_options {
public:
    llama3_options() {}
    llama3_options(const llama3_options&) = default;

    llama3_options
    head_dim(std::optional<std::size_t> head_dim) const noexcept;

    llama3_options
    n_heads(std::optional<std::size_t> n_heads) const noexcept;

    llama3_options
    n_kv_heads(std::optional<std::size_t> n_kv_heads) const noexcept;

    llama3_options
    n_layers(std::optional<std::size_t> n_layers) const noexcept;

    llama3_options
    max_seq_len(std::optional<std::size_t> max_seq_len) const noexcept;

    llama3_options
    heap_size(std::optional<std::size_t> heap_size) const noexcept;

    llama3_options
    rope_theta(std::optional<float> rope_theta) const noexcept;

    std::size_t
    head_dim() const noexcept;

    std::size_t
    n_heads() const noexcept;

    std::size_t
    n_kv_heads() const noexcept;

    std::size_t
    n_layers() const noexcept;

    std::size_t
    max_seq_len() const noexcept;

    std::size_t
    heap_size() const noexcept;

    float
    rope_theta() const noexcept;

private:
    std::size_t _M_head_dim = 0;
    std::size_t _M_n_heads = 0;
    std::size_t _M_n_kv_heads = 0;
    std::size_t _M_n_layers = 0;
    std::size_t _M_max_seq_len = 0;
    std::size_t _M_heap_size = 0;
    float _M_rope_theta = 0.0f;

    void
    set_head_dim(std::size_t head_dim);

    void
    set_n_heads(std::size_t n_heads);

    void
    set_n_kv_heads(std::size_t n_kv_heads);

    void
    set_n_layers(std::size_t n_layers);

    void
    set_max_seq_len(std::size_t max_seq_len);

    void
    set_heap_size(std::size_t heap_size);

    void
    set_rope_theta(float rope_theta);
};


llama3_options
default_llama3_1b_options();


/// Llama 3 is an auto-regressive language model that uses an optimized transformer architecture.
/// The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human
/// feedback (RLHF) to align with human preferences for helpfulness and safety.
template <
    typename T,
    contiguous_container Container = hardware_memory_container<T>,
    cache_t<T> Cache = sink_cache<T>>
class llama3 : public basic_layer {
private:
    using _Transformer = nn::transformer<T, Container>;
    using _TransformerArray = layer_array<_Transformer>;
    using _CacheArray = layer_array<Cache>;
    using _Embedding = nn::embedding<T, Container>;
    using _RMSNorm = nn::rmsnorm<T, Container>;
    using _Linear = nn::linear<T, Container>;

    _Embedding::layer_pointer _M_embedding;
    _RMSNorm::layer_pointer _M_norm;
    _Linear::layer_pointer _M_output;

    _TransformerArray::layer_pointer _M_transforms;
    _CacheArray::layer_pointer _M_caches;

    std::shared_ptr<basic_sampler<T>> _M_sampler;

public:
    using index_type = int32_t;
    using value_type = T;
    using container_type = Container;
    using cache_type = Cache;
    using tensor_type = future_tensor<index_type, 2>;

    using layer_type = llama3<T, Container, Cache>;
    using layer_pointer = shared_layer_ptr<layer_type>;

    /// Constructs a new Llama3 model with uninitialized weights with the given options.
    llama3(llama3_options options, hardware_accelerator& accelerator)
        requires cache_constructible<Cache>
    : basic_layer(accelerator),
      _M_sampler(std::make_shared<nucleus_sampler<value_type>>())
    {
        // The original implementation of Llama 3.2 shares the weight of token embeddings
        // and the output layer, use a shared tensor in order to reduce memory footprint.
        auto weight = shared_tensor(tensor<value_type, 2, container_type>());

        _M_embedding = register_layer("tok_embeddings", _Embedding(weight, accelerator));
        _M_norm = register_layer("norm", _RMSNorm(accelerator));
        _M_output = register_layer("output", _Linear(weight, accelerator));
        _M_transforms = register_layer("layers", _TransformerArray(accelerator));
        _M_caches = register_layer("caches", _CacheArray(accelerator));

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

    template <immutable_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);

        for (std::size_t i = 0; i < _M_transforms->size(); i++) {
            auto& transform = _M_transforms->at(i);
            auto& cache = _M_caches->at(i);

            x = transform(x, cache, start_pos);
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
};


} // namespace nn
} // namespace metalchat

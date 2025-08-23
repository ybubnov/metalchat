#pragma once

#include <format>
#include <list>
#include <optional>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/embedding.h>
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
    nn::shared_embedding<T, Container> _M_embedding;
    nn::shared_rmsnorm<T, Container> _M_norm;
    nn::shared_linear<T, Container> _M_output;

    std::vector<shared_transformer<T, Container>> _M_transforms;
    std::vector<Cache> _M_cache;

    std::shared_ptr<basic_sampler<T>> _M_sampler;

public:
    using index_type = int32_t;
    using value_type = T;
    using cache_type = Cache;
    using tensor_type = future_tensor<index_type, 2>;

    /// Constructs a new Llama3 model with uninitialized weights with the given options.
    llama3(llama3_options options, hardware_accelerator accelerator)
        requires cache_constructible<Cache>
    : basic_layer(accelerator),
      _M_sampler(std::make_shared<nucleus_sampler<value_type>>())
    {
        _M_embedding = register_layer("tok_embeddings", nn::embedding<T, Container>(accelerator));
        _M_norm = register_layer("norm", nn::rmsnorm<T, Container>(accelerator));
        _M_output = register_layer("output", nn::linear<T, Container>(accelerator));

        using layer_type = nn::transformer<T, Container>;

        const auto caching_opts = caching_options{
            .head_dim = options.head_dim(),
            .n_heads = options.n_heads(),
            .n_kv_heads = options.n_kv_heads(),
            .max_seq_len = options.max_seq_len(),
            .max_batch_size = 1
        };

        auto attention_opts = attention_options{
            .head_dim = options.head_dim(),
            .n_heads = options.n_heads(),
            .n_kv_heads = options.n_kv_heads(),
            .max_seq_len = options.max_seq_len(),
            .rope_theta = options.rope_theta()
        };

        for (std::size_t i = 0; i < options.n_layers(); i++) {
            auto layer_name = std::format("layers.{}", i);
            auto layer_value = layer_type(attention_opts, accelerator);
            auto layer_ptr = register_layer(layer_name, std::move(layer_value));

            _M_transforms.push_back(layer_ptr);
            _M_cache.emplace_back(caching_opts, accelerator);
        }
    }

    template <immutable_tensor2_t<index_type> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _M_embedding(input);

        for (std::size_t i = 0; i < _M_transforms.size(); i++) {
            auto& transform = _M_transforms[i];
            auto& cache = _M_cache[i];

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

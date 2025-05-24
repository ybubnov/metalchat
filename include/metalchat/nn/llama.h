#pragma once

#include <format>
#include <list>
#include <optional>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace nn {


template <
    typename T,
    contiguous_container Container = hardware_memory_container<T>,
    cache_t<T> Cache = sink_cache<T>>
class llama : public layer {
private:
    nn::shared_embedding<T, Container> _m_embedding;
    nn::shared_rmsnorm<T, Container> _m_norm;
    nn::shared_linear<T, Container> _m_output;

    std::vector<shared_transformer<T, Container>> _m_transforms;
    std::vector<Cache> _m_cache;

public:
    using value_type = T;
    using cache_type = Cache;
    using result_type = future_tensor<value_type, 3>;

    llama(std::size_t nlayers, attention_options& options, hardware_accelerator gpu)
        requires cache_constructible<Cache>
    : layer(gpu)
    {
        _m_embedding = register_layer("tok_embeddings", nn::embedding<T, Container>(gpu));
        _m_norm = register_layer("norm", nn::rmsnorm<T, Container>(gpu));
        _m_output = register_layer("output", nn::linear<T, Container>(gpu));

        using layer_type = nn::transformer<T, Container>;

        const auto cache_opts = cache_options{
            .head_dim = options.head_dim,
            .n_heads = options.n_heads,
            .n_kv_heads = options.n_kv_heads,
            .max_seq_len = options.max_seq_len,
            .max_batch_size = 1
        };

        for (std::size_t i = 0; i < nlayers; i++) {
            auto layer_name = std::format("layers.{}", i);
            auto layer_value = layer_type(options, gpu);
            auto layer_ptr = register_layer(layer_name, std::move(layer_value));

            _m_transforms.push_back(layer_ptr);
            _m_cache.emplace_back(cache_opts, gpu);
        }
    }

    template <immutable_tensor2_t<int32_t> Input>
    result_type
    operator()(Input input, std::size_t start_pos = 0)
    {
        auto x = _m_embedding(input);

        for (std::size_t i = 0; i < _m_transforms.size(); i++) {
            auto& transform = _m_transforms[i];
            auto& cache = _m_cache[i];

            x = transform(x, cache, start_pos);
        }

        auto output = _m_norm(x);

        auto len = output.size(1);
        output = output.narrow(1, len - 1, 1);

        return _m_output(output);
    }
};


} // namespace nn
} // namespace metalchat

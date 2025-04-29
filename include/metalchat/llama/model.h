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
#include <metalchat/llama/transformer.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace llama {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class model : public layer {
private:
    nn::embedding<T, hardware_memory_container<T>> _m_embedding;
    nn::rmsnorm<T, Container> _m_norm;
    nn::linear<T, hardware_memory_container<T>> _m_output;

    std::list<transformer<T, Container>> _m_transforms;
    std::reference_wrapper<hardware_accelerator> _m_gpu;

    auto
    create_additive_causal_mask(const std::size_t size) const
    {
        std::optional<shared_tensor<T, 2, hardware_memory_container<T>>> mask;

        if (size > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());
            auto m = full<T>({size, size}, -infinity, _m_gpu.get().get_allocator());
            triu(m);

            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

public:
    model(model&&) = default;
    model(const model&) = delete;

    model(std::size_t nlayers, attention_options& options, hardware_accelerator& gpu)
    : layer(),
      _m_embedding(gpu),
      _m_norm(gpu),
      _m_output(gpu),
      _m_transforms(),
      _m_gpu(gpu)
    {
        register_layer("tok_embeddings", _m_embedding);
        register_layer("norm", _m_norm);
        register_layer("output", _m_output);

        for (std::size_t i = 0; i < nlayers; i++) {
            _m_transforms.emplace_back(options, gpu);
            register_layer(std::format("layers.{}", i), _m_transforms.back());
        }
    }

    template <immutable_tensor2_t<int32_t> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        const auto mask = create_additive_causal_mask(input.size(1));
        auto x = _m_embedding(input);

        for (auto& transform : _m_transforms) {
            x = transform(x, mask, start_pos);
        }

        auto output = _m_norm(x);

        using s = indexing::slice;
        auto seqlen = output.size(1);
        output = output[s(), s(seqlen - 1, seqlen), s()];

        return _m_output(output);
    }
};


} // namespace llama
} // namespace metalchat

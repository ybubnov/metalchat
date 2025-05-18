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
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace nn {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class llama : public layer {
private:
    nn::shared_embedding<T, Container> _m_embedding;
    nn::shared_rmsnorm<T, Container> _m_norm;
    nn::shared_linear<T, Container> _m_output;

    std::list<shared_transformer<T, Container>> _m_transforms;

    auto
    create_additive_causal_mask(std::size_t size) const
    {
        std::optional<shared_tensor<T, 2, hardware_memory_container<T>>> mask;

        if (size > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());
            auto m = full<T>({size, size}, -infinity, accelerator().get_allocator());
            triu(m);

            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

public:
    using value_type = T;
    using result_type = future_tensor<value_type, 3>;

    llama(std::size_t nlayers, attention_options& options, hardware_accelerator gpu)
    : layer(gpu)
    {
        _m_embedding = register_layer("tok_embeddings", nn::embedding<T, Container>(gpu));
        _m_norm = register_layer("norm", nn::rmsnorm<T, Container>(gpu));
        _m_output = register_layer("output", nn::linear<T, Container>(gpu));

        using layer_type = nn::transformer<T, Container>;

        for (std::size_t i = 0; i < nlayers; i++) {
            auto layer_ptr = register_layer(std::format("layers.{}", i), layer_type(options, gpu));
            _m_transforms.push_back(layer_ptr);
        }
    }

    template <immutable_tensor2_t<int32_t> Input>
    result_type
    operator()(Input input, std::size_t start_pos = 0)
    {
        const auto mask = create_additive_causal_mask(input.size(1));
        auto x = _m_embedding(input);

        for (auto& transform : _m_transforms) {
            x = transform(x, mask, start_pos);
        }

        auto output = _m_norm(x);

        auto seqlen = output.size(1);
        output = output.narrow(1, seqlen - 1, 1);

        return _m_output(output);
    }

    void
    print()
    {
        std::cout << "llama!!" << std::endl;
    }
};


} // namespace nn
} // namespace metalchat

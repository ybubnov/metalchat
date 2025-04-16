#pragma once

#include <optional>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/llama/transformer.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace llama {


template <typename T, ContiguousContainer Container> class model {
private:
    nn::embedding<T, device_ref<T>> _m_embedding;
    nn::rmsnorm<T, Container> _m_norm;
    nn::linear<T, device_ref<T>> _m_output;

    std::vector<transformer<T, Container>> _m_layers;

    auto
    create_additive_causal_mask(const std::size_t size) const
    {
        std::optional<shared_tensor<T, 2, owning_ref<T>>> mask;

        if (size > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());
            auto m = full<T>({size, size}, -infinity);
            triu(m);

            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

public:
    model(model&&) = default;

    model(
        nn::embedding<T, device_ref<T>>&& embedding,
        nn::rmsnorm<T, Container>&& norm,
        nn::linear<T, device_ref<T>>&& output,
        std::vector<transformer<T, Container>>&& layers
    )
    : _m_embedding(std::move(embedding)),
      _m_norm(std::move(norm)),
      _m_output(std::move(output)),
      _m_layers(std::move(layers))
    {}

    template <immutable_tensor2d InputTensor> requires(integral<typename InputTensor::value_type>)
    auto
    operator()(InputTensor input, std::size_t start_pos = 0)
    {
        const auto mask = create_additive_causal_mask(input.size(1));
        auto x = _m_embedding(input);

        for (auto& layer : _m_layers) {
            x = layer(x, mask, start_pos);
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

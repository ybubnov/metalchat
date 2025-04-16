#pragma once

#include <vector>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/llama/transformer.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace llama {


template <typename T, ContiguousContainer Container> class model {
private:
    nn::embedding<T, Container> _m_embedding;
    nn::rmsnorm<T, Container> _m_norm;
    nn::linear<T, Container> _m_output;

    std::vector<transformer<T, Container>> _m_layers;

public:
    model(
        nn::embedding<T, Container> embedding,
        nn::rmsnorm<T, Container> norm,
        nn::linear<T, Container> output,
        const std::vector<transformer<T, Container>> layers
    )
    : _m_embedding(embedding),
      _m_norm(norm),
      _m_output(output),
      _m_layers(layers)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 2, InputContainer>& input)
    {
        auto mask = full({input.size(1), input.size(1)}, -1e9);
        triu(mask);

        auto emb = _m_embedding(input);
        for (const auto& l : _m_layers) {
            input = l(input, mask);
        }

        input = _m_norm(input);
        return _m_output(input); // _m_output(input[:, -1]) - take only the last dimension;
    }
};


} // namespace llama
} // namespace metalchat

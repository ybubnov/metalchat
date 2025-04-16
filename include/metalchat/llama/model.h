#pragma once

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
    nn::embedding<T, Container> _m_embedding;
    nn::rmsnorm<T, Container> _m_norm;
    nn::linear<T, Container> _m_output;

    std::vector<transformer<T, Container>> _m_layers;

public:
    model(model&&) = default;

    model(
        nn::embedding<T, Container>&& embedding,
        nn::rmsnorm<T, Container>&& norm,
        nn::linear<T, Container>&& output,
        std::vector<transformer<T, Container>>&& layers
    )
    : _m_embedding(std::move(embedding)),
      _m_norm(std::move(norm)),
      _m_output(std::move(output)),
      _m_layers(std::move(layers))
    {}

    template <integral IndexType, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<IndexType, 2, InputContainer>& input)
    {
        auto mask = full<T>({input.size(1), input.size(1)}, -1e9);
        triu(mask);

        auto x = _m_embedding(input);

        for (auto& layer : _m_layers) {
            x = layer(x, mask);
        }

        auto output = _m_norm(x);
        // output (bs, len, 128256).
        return _m_output(output); // _m_output(input[:, -1]) - take only the last dimension;
    }
};


} // namespace llama
} // namespace metalchat

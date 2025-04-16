#pragma once


#include <metalchat/kernel/embedding.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer Container> class embedding {
private:
    const tensor<T, 2, Container> _m_weight;
    metalchat::embedding<T> _m_embedding;

public:
    embedding(const tensor<T, 2, Container> weight, device& device)
    : _m_weight(weight),
      _m_embedding(device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<int32_t, 1, InputContainer>& input)
    {
        return _m_embedding(input, _m_weight);
    }
};


} // namespace nn
} // namespace metalchat

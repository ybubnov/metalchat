#pragma once

#include <iostream>

#include <metalchat/functional.h>
#include <metalchat/kernel/embedding.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer Container> class embedding {
private:
    tensor<T, 2, Container> _m_weight;
    metalchat::embedding<T> _m_embedding;

public:
    embedding(embedding&&) = default;

    embedding(tensor<T, 2, Container>&& weight, device& device)
    : _m_weight(std::move(weight)),
      _m_embedding(device)
    {}

    template <integral IndexType, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<IndexType, 2, InputContainer>& input)
    {
        return _m_embedding(input, _m_weight);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const embedding& e)
    {
        os << "nn::embedding<" << type_traits<T>::name() << ">";
        os << "(" << e._m_weight.size(0) << ", " << e._m_weight.size(1) << ")";
        return os;
    }
};


} // namespace nn
} // namespace metalchat

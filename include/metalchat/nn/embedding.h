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
    operator()(const tensor<IndexType, 1, InputContainer>& input)
    {
        return _m_embedding(input, _m_weight);
    }

    template <integral IndexType, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<IndexType, 2, InputContainer>& input)
    {
        // TODO: rework it to use a single kernel.
        using tensor_type = tensor<T, 2, device_ref<T>>;

        std::vector<tensor_type> outputs;
        for (auto batch = 0; batch < input.size(0); batch++) {
            outputs.push_back(_m_embedding(input[batch], _m_weight));
        }

        auto output = concatenate(outputs.begin(), outputs.end(), 0);
        return output.reshape({int(input.size(0)), int(input.size(1)), -1});
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

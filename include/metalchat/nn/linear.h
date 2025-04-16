#pragma once


#include <metalchat/kernel/sgemm.h>


namespace metalchat {
namespace nn {


template <typename T, ContiguousContainer WeightContainer> class linear {
private:
    const tensor<T, 2, WeightContainer>& m_weight;
    sgemm<T> m_sgemm;

public:
    linear(const tensor<T, 2, WeightContainer>& weight, device& device)
    : m_weight(weight),
      m_sgemm(device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 2, InputContainer>& input)
    {
        return m_sgemm(input, m_weight.t());
    }
};


} // namespace nn
} // namespace metalchat

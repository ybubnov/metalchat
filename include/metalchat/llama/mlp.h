#pragma once


#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


template <typename T, ContiguousContainer Container> class mlp {
private:
    nn::linear<T, Container> m_gate_proj;
    nn::linear<T, Container> m_up_proj;
    nn::linear<T, Container> m_down_proj;

public:
    mlp(const tensor<T, 2, Container>& gate_proj_weight,
        const tensor<T, 2, Container>& up_proj_weight,
        const tensor<T, 2, Container>& down_proj_weight,
        device& device)
    : m_gate_proj(gate_proj_weight, device),
      m_up_proj(up_proj_weight, device),
      m_down_proj(down_proj_weight, device)
    {}

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 2, InputContainer>& input)
    {
        return m_down_proj(m_up_proj(input));
    }
};


} // namespace llama
} // namespace metalchat

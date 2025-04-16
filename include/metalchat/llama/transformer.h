#pragma once

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/llama/attention.h>
#include <metalchat/llama/feed_forward.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace llama {

template <typename T, ContiguousContainer Container> class transformer {
private:
    attention<T, Container> _m_attention;
    nn::rmsnorm<T, Container> _m_attention_norm;

    feed_forward<T, Container> _m_ff;
    nn::rmsnorm<T, Container> _m_ff_norm;

    sum<T> _m_sum;

public:
    transformer(
        const attention<T, Container>& attention,
        const nn::rmsnorm<T, Container>& attention_norm,
        const feed_forward<T, Container>& ff,
        const nn::rmsnorm<T, Container>& ff_norm,
        device& device,
    )
    : _m_attention(attention),
      _m_attention_norm(attention_norm),
      _m_ff(ff),
      _m_ff_norm(ff_norm),
      _m_sum(device),
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer MaskContainer>
    auto
    operator(const tensor<T, 3, InputContainer>& input, const tensor<T, 2, MaksContainer>& mask)
    {
        auto r = _m_attention(_m_attention_norm(input), mask);
        auto h = _m_sum(input, r);

        r = _m_ff(_m_ff_norm(h));
        auto output = _m_sum(h, r);
        return output;
    }
};

} // namespace llama
} // namespace metalchat

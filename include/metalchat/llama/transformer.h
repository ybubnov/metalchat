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
    transformer(transformer&&) = default;
    transformer(const transformer&) = delete;

    transformer(
        attention<T, Container>&& attention,
        nn::rmsnorm<T, Container>&& attention_norm,
        feed_forward<T, Container>&& ff,
        nn::rmsnorm<T, Container>&& ff_norm,
        device& device
    )
    : _m_attention(std::move(attention)),
      _m_attention_norm(std::move(attention_norm)),
      _m_ff(std::move(ff)),
      _m_ff_norm(std::move(ff_norm)),
      _m_sum(device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer MaskContainer>
    auto
    operator()(const tensor<T, 3, InputContainer>& input, const tensor<T, 2, MaskContainer>& mask)
    {
        std::cout << ">>>transformer" << std::endl;
        auto norm = _m_attention_norm(input);

        auto r = _m_attention(norm, mask);
        auto h = _m_sum(input, r);

        r = _m_ff(_m_ff_norm(h));
        auto output = _m_sum(h, r);
        std::cout << "<<<transformer" << std::endl;
        return output;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const transformer& t)
    {
        os << "llama::transformer<" << type_traits<T>::name() << ">()";
        return os;
    }
};

} // namespace llama
} // namespace metalchat

#pragma once

#include <cmath>
#include <optional>
#include <ranges>

#include <metalchat/functional.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


struct attention_options {
    std::size_t head_dim;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    float base;

    inline int
    repeats()
    {
        return n_heads / n_kv_heads;
    }
};


template <typename T, ContiguousContainer Container> class attention {
private:
    nn::linear<T, Container> m_wq;
    nn::linear<T, Container> m_wk;
    nn::linear<T, Container> m_wv;
    nn::linear<T, Container> m_wo;

    rope<T> m_rope;
    scalar_mul<T> m_mul;
    bmm<T> m_matmul;
    sum<T> m_sum;
    softmax<T> m_softmax;

    attention_options m_options;
    float m_scale;

public:
    attention(attention&&) = default;
    attention(const attention&) = default;

    attention(
        tensor<T, 2, Container>&& wq,
        tensor<T, 2, Container>&& wk,
        tensor<T, 2, Container>&& wv,
        tensor<T, 2, Container>&& wo,
        attention_options& options,
        device& device
    )
    : m_wq(std::move(wq), device),
      m_wk(std::move(wk), device),
      m_wv(std::move(wv), device),
      m_wo(std::move(wo), device),
      m_rope(device, options.head_dim, /*base=*/options.base),
      m_mul(device),
      m_matmul(device),
      m_sum(device),
      m_softmax(device),
      m_options(options),
      m_scale(std::pow(options.head_dim, -0.5))
    {}


    template <ContiguousContainer InputContainer, ContiguousContainer MaskContainer>
    auto
    operator()(const tensor<T, 3, InputContainer>& input, const tensor<T, 2, MaskContainer>& mask)
    {
        int bs = input.size(0);
        int len = input.size(1);
        int n_heads = m_options.n_heads;
        int n_kv_heads = m_options.n_kv_heads;

        auto queries = m_wq(input).reshape({bs, len, n_heads, -1}).transpose({0, 2, 1, 3});
        auto keys = m_wk(input).reshape({bs, len, n_kv_heads, -1}).transpose({0, 2, 1, 3});
        auto values = m_wv(input).reshape({bs, len, n_kv_heads, -1}).transpose({0, 2, 1, 3});

        // TODO: cache queries and keys.
        //
        // TODO: does it even make sense to make the repetition, would it be better
        // to adjust the implementation of the "rope", so it uses the same memory?
        std::cout << "K sizes=" << keys.sizes() << std::endl;
        auto keys_rep = repeat_interleave(std::move(keys), m_options.repeats(), /*dim=*/2);
        std::cout << "K shape=" << keys_rep.sizes() << std::endl;
        auto K = keys_rep.reshape({bs, n_heads, len, -1});
        auto K_ = m_rope(K);

        auto values_rep = repeat_interleave(std::move(values), m_options.repeats(), /*dim=*/2);
        auto V = values_rep.reshape({bs, n_heads, len, -1});
        auto V_ = m_rope(V);

        auto scores = m_matmul(m_mul(queries, m_scale), K_.transpose({0, 1, 3, 2}));
        // auto scores_ = m_sum(scores, mask);

        auto scores_ = m_softmax(scores);
        auto output = m_matmul(scores_, V_).transpose({0, 2, 1, 3}).reshape({bs, len, -1});

        return m_wo(output);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const attention& a)
    {
        os << "llama::attention<" << type_traits<T>::name() << ">()";
        return os;
    }
};


} // namespace llama
} // namespace metalchat

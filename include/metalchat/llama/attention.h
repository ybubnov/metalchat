#pragma once

#include <cmath>

#include <metalchat/functional/embedding.h>
#include <metalchat/functional/sgemm.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace llama {


struct attention_options {
    std::size_t head_dim;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    float base;
};


template <typename T, ContiguousContainer Container> class attention {
private:
    nn::linear<T, Container> m_wq;
    nn::linear<T, Container> m_wk;
    nn::linear<T, Container> m_wv;
    nn::linear<T, Container> m_wo;

    rope<T> m_rope;
    scalar_mul<T> m_mul;
    sgemm<T> m_matmul;

    attention_options m_options;
    float m_scale;

public:
    attention(
        const tensor<T, 2, Container>& wq,
        const tensor<T, 2, Container>& wk,
        const tensor<T, 2, Container>& wv,
        const tensor<T, 2, Container>& wo,
        const attention_options& options,
        device& device
    )
    : m_wq(wq, device),
      m_wk(wk, device),
      m_wv(wv, device),
      m_wo(wo, device),
      m_rope(device, options.head_dim, /*base=*/options.base, /*scale=?*/),
      m_mul(device),
      m_options(options),
      m_scale(std::pow(options.head_dim, -0.5))
    {}

    template <ContiguousContainer InputContainer>
    void
    operator()(const tensor<T, 3, InputContainer>& input)
    {
        auto bs = input.size(0);
        auto len = input.size(1);

        auto queries = m_wq(input);
        auto keys = m_wk(input);
        auto values = m_wv(input);

        queries = queries.reshape({bs, len, m_options.n_heads, -1}).transpose({0, 2, 1, 3});
        keys = keys.reshape({bs, len, m_options.n_kv_heads, -1}).transpose({0, 2, 1, 3});
        values = values.reshape({bs, len, m_options.n_kv_heads, -1}).transpose({0, 2, 1, 3});

        // TODO: repeat + concatenate queries and keys.
        queries = m_rope(queries);
        keys = m_rope(keys);

        auto scores = m_matmul(m_mul(queries, m_scale), keys.transpose({0, 1, 3, 2}));
        // scores = m_sum(scores, mask);

        // TODO: implement softmax over -1 dimension.
        scores = m_softmax(scores, /*dim=-1*/);
        auto output = m_matmul(scores, values).transpose({0, 2, 1, 3}).reshape({bs, len, -1});
        output = m_wo(output);

        return output;
    }
};


} // namespace llama
} // namespace metalchat

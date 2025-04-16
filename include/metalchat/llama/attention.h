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

    inline std::size_t
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
    sum2<T> m_sum;
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
      m_scale(1.0 / std::sqrt(float(options.head_dim)))
    {}


    template <ContiguousContainer InputContainer, ContiguousContainer MaskContainer>
    auto
    operator()(const tensor<T, 3, InputContainer>& input, const tensor<T, 2, MaskContainer>& mask)
    {
        int bs = input.size(0);
        int len = input.size(1);
        int n_heads = m_options.n_heads;
        int n_kv_heads = m_options.n_kv_heads;
        auto n_reps = m_options.repeats();

        auto q = m_wq(input).reshape({bs, len, n_heads, -1}).transpose({0, 2, 1, 3});
        auto k = m_wk(input).reshape({bs, len, n_kv_heads, -1}).transpose({0, 2, 1, 3});
        auto v = m_wv(input).reshape({bs, len, n_kv_heads, -1}).transpose({0, 2, 1, 3});

        auto repeat_kv = [&](tensor<T, 4, device_ref<T>>&& t) -> auto {
            auto reps = repeat_interleave(std::move(t), n_reps, /*dim=*/2);
            return reps.reshape({bs, n_heads, len, -1});
        };

        auto keys = repeat_kv(std::move(k));
        auto values = repeat_kv(std::move(v));

        // TODO: cache queries and keys.
        auto q_rot = m_rope(q);
        auto k_rot = m_rope(keys);

        auto scores = m_matmul(m_mul(q_rot, m_scale), k_rot.transpose({0, 1, 3, 2}));
        auto scores_ = m_sum(scores, mask);

        scores_ = m_softmax(scores_);
        auto output = m_matmul(scores_, values).transpose({0, 2, 1, 3}).reshape({bs, len, -1});

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

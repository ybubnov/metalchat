#pragma once

#include <cmath>
#include <optional>
#include <ranges>

#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/layer.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/linear.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace nn {


struct attention_options {
    std::size_t head_dim;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t max_seq_len;
    std::size_t max_batch_size;
    float rope_theta;

    inline std::size_t
    repeats() const
    {
        return n_heads / n_kv_heads;
    }

    inline float
    scale() const
    {
        return 1.0 / std::sqrt(float(head_dim));
    }
};


template <typename T, contiguous_container Container> class attention : public layer {
private:
    static constexpr std::size_t input_size = 4;

    nn::shared_linear<T, Container> m_wq;
    nn::shared_linear<T, Container> m_wk;
    nn::shared_linear<T, Container> m_wv;
    nn::shared_linear<T, Container> m_wo;

    nn::rope<T> _m_rope;

    nn::attention_options _m_options;
    T _m_scale;

    kernel::cpy<T> _m_cpy;

    template <immutable_tensor_t<T> Input>
    auto
    contiguous(Input input, std::size_t dim)
    {
        auto output = future_tensor(empty_like<T>(input, accelerator().get_allocator()));

        for (std::size_t offset = 0; offset < output.size(dim); offset++) {
            auto future = _m_cpy(input.narrow(dim, offset, 1), output.narrow(dim, offset, 1));
            output = future_tensor(output, future);
        }

        return output;
    }

public:
    attention(
        attention_options& options, hardware_accelerator accelerator, std::size_t max_batch_size = 1
    )
    : layer(accelerator),
      _m_rope(options.head_dim, options.max_seq_len, /*thetha=*/options.rope_theta, accelerator),
      _m_options(options),
      _m_scale(options.scale()),
      _m_cpy(accelerator)
    {
        m_wq = register_layer("wq", nn::linear<T, Container>(accelerator));
        m_wk = register_layer("wk", nn::linear<T, Container>(accelerator));
        m_wv = register_layer("wv", nn::linear<T, Container>(accelerator));
        m_wo = register_layer("wo", nn::linear<T, Container>(accelerator));
    }

    template <immutable_tensor3_t<T> Input, cache_t<T> Cache>
    auto
    operator()(Input input, Cache& cache, std::size_t start_pos = 0)
    {
        int bs = input.size(0);
        int len = input.size(1);
        int n_heads = _m_options.n_heads;
        int n_kv_heads = _m_options.n_kv_heads;
        auto n_reps = _m_options.repeats();
        const int head_dim = _m_options.head_dim;

        auto q = m_wq(input).view({bs, len, n_heads, head_dim});
        auto k = m_wk(input).view({bs, len, n_kv_heads, head_dim});
        auto v = m_wv(input).view({bs, len, n_kv_heads, head_dim});

        q = _m_rope(q, /*start_pos=*/start_pos);
        k = _m_rope(k, /*start_pos=*/start_pos);

        auto [kk, vv, mask] = cache.update(k, v, start_pos);

        auto repeat_kv = [&]<immutable_tensor4_t<T> Tensor>(Tensor&& t) -> auto {
            int slen = t.size(1);
            auto reps = repeat_interleave(t, n_reps, /*dim=*/2, accelerator());
            return reps.view({bs, slen, n_heads, head_dim});
        };

        // shape: bs, cache + len, n_heads, head_dim.
        auto keys = repeat_kv(std::move(kk));
        auto values = repeat_kv(std::move(vv));

        auto queries = q.transpose({0, 2, 1, 3});
        keys = keys.transpose({0, 2, 3, 1});
        values = values.transpose({0, 2, 1, 3});

        auto scores = mul(matmul(queries, keys, accelerator()), _m_scale, accelerator());
        if (mask.has_value()) {
            scores = add2(scores, mask.value(), accelerator());
        }
        scores = softmax(scores, accelerator());

        auto output = matmul(scores, values, accelerator()).transpose({0, 2, 1, 3});
        output = contiguous(output, /*dim=*/1);

        return m_wo(output.view({bs, len, -1}));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const attention&)
    {
        os << "nn::attention<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container>
using shared_attention = shared_layer<attention<T, Container>>;


} // namespace nn
} // namespace metalchat

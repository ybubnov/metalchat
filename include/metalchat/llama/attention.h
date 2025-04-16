#pragma once

#include <cmath>
#include <optional>
#include <ranges>

#include <metalchat/functional.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/linear.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace llama {


struct attention_options {
    std::size_t head_dim;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t max_seq_len;
    float rope_theta;

    inline std::size_t
    repeats()
    {
        return n_heads / n_kv_heads;
    }
};


template <typename T, ContiguousContainer Container> class attention {
private:
    static constexpr std::size_t input_size = 4;

    nn::linear<T, Container> m_wq;
    nn::linear<T, Container> m_wk;
    nn::linear<T, Container> m_wv;
    nn::linear<T, Container> m_wo;

    nn::rope<T> m_rope;
    scalar_mul<T> m_mul;
    bmm<T> m_matmul;
    sum2<T> m_sum;
    softmax<T> m_softmax;

    attention_options m_options;
    float m_scale;

    shared_tensor<T, input_size, device_ref<T>> _m_cache_k;
    shared_tensor<T, input_size, device_ref<T>> _m_cache_v;

    cpy<T> _m_cpy;
    device& _m_device;

    template <immutable_tensor InputTensor>
    auto
    contiguous(InputTensor input, std::size_t dim)
    {
        auto output = future_tensor(empty_like(input, _m_device));

        for (std::size_t offset = 0; offset < output.size(dim); offset++) {
            auto future = _m_cpy(input.narrow(dim, offset, 1), output.narrow(dim, offset, 1));
            output = future_tensor(output, future);
        }

        return output;
    }

    template <immutable_tensor4d InputTensor, immutable_device_tensor4d CacheTensor>
    auto
    cache_copy(
        InputTensor input,
        CacheTensor cache,
        std::size_t bs,
        std::size_t start_pos,
        std::size_t size
    )
    {
        using s = indexing::slice;
        auto target = cache[s(0, bs), s(start_pos, start_pos + size), s(), s()];
        auto result = cache[s(0, bs), s(0, start_pos + size), s(), s()];

        return future_tensor(result, _m_cpy(input, target));
    }

    template <immutable_tensor4d InputTensor>
    inline auto
    cache_keys(InputTensor input, std::size_t bs, std::size_t begin, std::size_t size)
    {
        return cache_copy(input, _m_cache_k, bs, begin, size);
    }

    template <immutable_tensor4d InputTensor>
    inline auto
    cache_values(InputTensor input, std::size_t bs, std::size_t begin, std::size_t size)
    {
        return cache_copy(input, _m_cache_v, bs, begin, size);
    }

public:
    attention(attention&&) = default;
    attention(const attention&) = default;

    attention(
        tensor<T, 2, Container>&& wq,
        tensor<T, 2, Container>&& wk,
        tensor<T, 2, Container>&& wv,
        tensor<T, 2, Container>&& wo,
        attention_options& options,
        device& device,
        std::size_t max_batch_size = 1
    )
    : m_wq(std::move(wq), device),
      m_wk(std::move(wk), device),
      m_wv(std::move(wv), device),
      m_wo(std::move(wo), device),
      m_rope(options.head_dim, options.max_seq_len, /*thetha=*/options.rope_theta, device),
      m_mul(device),
      m_matmul(device),
      m_sum(device),
      m_softmax(device),
      m_options(options),
      m_scale(1.0 / std::sqrt(float(options.head_dim))),
      _m_cache_k(empty<T>(
          {max_batch_size, options.max_seq_len, options.n_kv_heads, options.head_dim}, device
      )),
      _m_cache_v(empty<T>(
          {max_batch_size, options.max_seq_len, options.n_kv_heads, options.head_dim}, device
      )),
      _m_cpy(device),
      _m_device(device)
    {}

    template <immutable_tensor3d InputTensor, immutable_tensor2d MaskTensor>
    auto
    operator()(InputTensor input, const std::optional<MaskTensor> mask, std::size_t start_pos = 0)
    {
        int bs = input.size(0);
        int len = input.size(1);
        int n_heads = m_options.n_heads;
        int n_kv_heads = m_options.n_kv_heads;
        auto n_reps = m_options.repeats();
        const int head_dim = m_options.head_dim;

        auto q = m_wq(input).view({bs, len, n_heads, head_dim});
        auto k = m_wk(input).view({bs, len, n_kv_heads, head_dim});
        auto v = m_wv(input).view({bs, len, n_kv_heads, head_dim});

        q = m_rope(q, /*start_pos=*/start_pos);
        k = m_rope(k, /*start_pos=*/start_pos);

        auto kk = cache_keys(k, bs, start_pos, len);
        auto vv = cache_values(v, bs, start_pos, len);

        auto repeat_kv = [&]<immutable_tensor4d Tensor>(Tensor&& t) -> auto {
            int slen = t.size(1);
            auto reps = repeat_interleave(t, n_reps, /*dim=*/2, _m_device);
            return reps.view({bs, slen, n_heads, head_dim});
        };

        // shape: bs, cache + len, n_heads, head_dim.
        auto keys = repeat_kv(std::move(kk));
        auto values = repeat_kv(std::move(vv));

        auto queries = q.transpose({0, 2, 1, 3});
        keys = keys.transpose({0, 2, 3, 1});
        values = values.transpose({0, 2, 1, 3});

        auto scores = m_mul(m_matmul(queries, keys), m_scale);

        if (mask.has_value()) {
            scores = m_sum(scores, mask.value());
        }
        scores = m_softmax(scores);

        auto output = m_matmul(scores, values).transpose({0, 2, 1, 3});
        output = contiguous(output, /*dim=*/1);

        return m_wo(output.view({bs, len, -1}));
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

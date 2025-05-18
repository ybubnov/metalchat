#pragma once

#include <cmath>
#include <optional>
#include <ranges>

#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/layer.h>
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

    future_tensor<T, input_size> _m_cache_k;
    future_tensor<T, input_size> _m_cache_v;

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

    auto
    cache_alloc(const attention_options& options, std::size_t max_batch_size)
    {
        return future_tensor(empty<T>(
            {max_batch_size, options.max_seq_len, options.n_kv_heads, options.head_dim},
            accelerator().get_allocator()
        ));
    }

    /// Cache the intermediate results (Rotation Positional Encodings) into the cache tensor.
    ///
    /// The implementation allows to store the inference results for the position larger than
    /// the cache size: it simply drops the least recent results. It works like a sliding window.
    ///
    /// The implementation does not track if the specified start position corresponds to the
    /// latest used start position. So if user called an attention layer with `start_pos = 15`
    /// with cache size = 16, and then makes the next call with `start_pos = 44`, model won't
    /// complain, but the result is not won't be correct.
    template <immutable_tensor4_t<T> Input, immutable_hardware_tensor4_t<T> Cache>
    auto
    cache_copy(Input input, Cache cache, std::size_t bs, std::size_t start_pos, std::size_t len)
    {
        using s = indexing::slice;
        auto batch_size = cache.size(0);
        auto cache_size = cache.size(1);
        std::size_t prefix_len = 4;

        if (len > cache_size) {
            throw std::invalid_argument(std::format(
                "nn::attention: requested length ({}) is larger than the cache size ({})", len,
                cache_size
            ));
        }

        // When the cache is full (meaning that start position spilled over the boundaries
        // of the cache), rotate it left and store the inferred results into the right-most
        // position.
        if (start_pos >= cache_size) {
            auto cache_new = cache_alloc(_m_options, batch_size);
            cache_new = future_tensor(
                cache_new,
                _m_cpy(cache.narrow(1, 0, prefix_len), cache_new.narrow(1, 0, prefix_len))
            );

            auto rolled_cache = roll(
                cache.narrow(1, prefix_len, cache_size - prefix_len),
                cache_new.narrow(1, prefix_len, cache_size - prefix_len),
                /*shift=*/int32_t(len), /*dim=*/1, accelerator()
            );

            cache = future_tensor(cache_new, rolled_cache);
            start_pos = cache_size - len;
        }

        // Write the result of computation into the "target" tensor, so we could reuse
        // it on the next iteration again. To make precise inference, model will use all
        // previously cached results (or up to the end position).
        auto end_pos = start_pos + len;
        auto target = cache[s(0, bs), s(start_pos, end_pos), s(), s()];

        cache = future_tensor(cache, _m_cpy(input, target));
        auto cached_data = cache[s(0, bs), s(0, end_pos), s(), s()];

        return std::make_tuple(cache, cached_data);
    }

    template <immutable_tensor4_t<T> Input>
    inline auto
    cache_keys(Input input, std::size_t bs, std::size_t start_pos, std::size_t len)
    {
        auto [cache, keys] = cache_copy(input, _m_cache_k, bs, start_pos, len);
        _m_cache_k = cache;
        return keys;
    }

    template <immutable_tensor4_t<T> Input>
    inline auto
    cache_values(Input input, std::size_t bs, std::size_t start_pos, std::size_t len)
    {
        auto [cache, values] = cache_copy(input, _m_cache_v, bs, start_pos, len);
        _m_cache_v = cache;
        return values;
    }

public:
    attention(
        attention_options& options, hardware_accelerator accelerator, std::size_t max_batch_size = 1
    )
    : layer(accelerator),
      _m_rope(options.head_dim, options.max_seq_len, /*thetha=*/options.rope_theta, accelerator),
      _m_options(options),
      _m_scale(options.scale()),
      _m_cache_k(cache_alloc(options, max_batch_size)),
      _m_cache_v(cache_alloc(options, max_batch_size)),
      _m_cpy(accelerator)
    {
        m_wq = register_layer("wq", nn::linear<T, Container>(accelerator));
        m_wk = register_layer("wk", nn::linear<T, Container>(accelerator));
        m_wv = register_layer("wv", nn::linear<T, Container>(accelerator));
        m_wo = register_layer("wo", nn::linear<T, Container>(accelerator));

        // register_parameter("cache_k", _m_cache_k.get().get());
        // register_parameter("cache_v", _m_cache_v.get().get());
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, const std::optional<Mask> mask, std::size_t start_pos = 0)
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

        auto kk = cache_keys(k, bs, start_pos, len);
        auto vv = cache_values(v, bs, start_pos, len);

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

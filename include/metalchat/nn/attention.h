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


template <typename T, contiguous_container Container> class attention : public basic_layer {
private:
    static constexpr std::size_t input_size = 4;

    nn::shared_linear<T, Container> _M_wq;
    nn::shared_linear<T, Container> _M_wk;
    nn::shared_linear<T, Container> _M_wv;
    nn::shared_linear<T, Container> _M_wo;

    nn::rope<T> _M_rope;

    nn::attention_options _M_options;
    T _M_scale;

    kernel::clone<T> _M_clone;

    template <immutable_tensor_t<T> Input>
    auto
    contiguous(Input input, std::size_t dim)
    {
        auto output = future_tensor(empty_like<T>(input, accelerator().get_allocator()));

        for (std::size_t offset = 0; offset < output.size(dim); offset++) {
            auto future = _M_clone(input.narrow(dim, offset, 1), output.narrow(dim, offset, 1));
            output = future_tensor(output, future);
        }

        return output;
    }

public:
    attention(
        attention_options& options, hardware_accelerator accelerator, std::size_t max_batch_size = 1
    )
    : basic_layer(accelerator),
      _M_rope(options.head_dim, options.max_seq_len, /*thetha=*/options.rope_theta, accelerator),
      _M_options(options),
      _M_scale(options.scale()),
      _M_clone(accelerator)
    {
        _M_wq = register_layer("wq", nn::linear<T, Container>(accelerator));
        _M_wk = register_layer("wk", nn::linear<T, Container>(accelerator));
        _M_wv = register_layer("wv", nn::linear<T, Container>(accelerator));
        _M_wo = register_layer("wo", nn::linear<T, Container>(accelerator));
    }

    template <immutable_tensor3_t<T> Input, cache_t<T> Cache>
    auto
    operator()(Input input, Cache& cache, std::size_t start_pos = 0)
    {
        int bs = input.size(0);
        int len = input.size(1);
        int n_heads = _M_options.n_heads;
        int n_kv_heads = _M_options.n_kv_heads;
        auto n_reps = _M_options.repeats();
        const int head_dim = _M_options.head_dim;

        auto q = _M_wq(input).view({bs, len, n_heads, head_dim});
        auto k = _M_wk(input).view({bs, len, n_kv_heads, head_dim});
        auto v = _M_wv(input).view({bs, len, n_kv_heads, head_dim});

        q = _M_rope(q, /*start_pos=*/start_pos);
        k = _M_rope(k, /*start_pos=*/start_pos);

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

        auto scores = mul(matmul(queries, keys, accelerator()), _M_scale, accelerator());
        if (mask.has_value()) {
            scores = add2(scores, mask.value(), accelerator());
        }
        scores = softmax(scores, accelerator());

        auto output = matmul(scores, values, accelerator()).transpose({0, 2, 1, 3});
        output = contiguous(output, /*dim=*/1);

        return _M_wo(output.view({bs, len, -1}));
    }

    friend std::ostream&
    operator<<(std::ostream& os, const attention&)
    {
        os << "nn::attention<" << type_traits<T>::name() << ">()";
        return os;
    }
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
using shared_attention = shared_layer_ptr<attention<T, Container>>;


} // namespace nn
} // namespace metalchat

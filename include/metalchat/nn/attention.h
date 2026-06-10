// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <optional>
#include <ranges>

#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer.h>
#include <metalchat/nn/linear.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace nn {


struct attention_options {
    /// Per-attention head embedding dimension.
    std::size_t head_dim;
    /// Number of query heads.
    std::size_t n_heads;
    /// Number of key and value heads.
    std::size_t n_kv_heads;
    /// Maximum sequence length model will be run with.
    std::size_t max_seq_len;
    /// Batch size the model will be run with.
    std::size_t max_batch_size;
    float rope_theta;

    /// Scaling factor applied prior to softmax.
    float scale;

    /// Enables RMS-normalization to queries and values, when the value is provided.
    ///
    /// \note The layer won't even register RMS-normalization layers, when the value
    /// is empty.
    std::optional<float> norm_eps = std::nullopt;

    /// When RMS is distributed as residuals from a mean, then this parameter is used
    /// to restore the weight of RMS to the original state.
    ///
    /// \note This option has no effect on attention normalization, when `norm_eps` is unset.
    std::optional<float> norm_mu = std::nullopt;

    inline std::size_t
    repeats() const
    {
        return n_heads / n_kv_heads;
    }
};


/// Allows the model to jointly attend to information from different representation subspaces.
///
/// This \ref attention layer implements the original architecture described in the
/// <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need paper</a>.
template <typename T, contiguous_container Container, cache_t<T> Cache = sink_cache<T>>
class attention : public basic_layer {
private:
    static constexpr std::size_t input_size = 4;

    using BasicLinear = nn::basic_linear<T, Container>;
    using Linear = nn::linear<T, Container>;
    using RMSNorm = nn::rmsnorm<T, Container>;
    using RotaryPositionalEmbedding = nn::rope<T>;

    polymorphic_layer<BasicLinear> _M_wq;
    polymorphic_layer<BasicLinear> _M_wk;
    polymorphic_layer<BasicLinear> _M_wv;
    polymorphic_layer<BasicLinear> _M_wo;

    indirect_layer<RotaryPositionalEmbedding> _M_rope;
    indirect_layer<RMSNorm> _M_wq_norm;
    indirect_layer<RMSNorm> _M_wk_norm;
    indirect_layer<Cache> _M_cache;

    nn::attention_options _M_options;
    kernel::clone<T> _M_clone;

    T _M_scale;

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
    using value_type = T;
    using container_type = Container;

    attention(
        const attention_options& options, const indirect_layer<RotaryPositionalEmbedding>& rope
    )
    : basic_layer(rope.accelerator()),
      _M_rope(rope),
      _M_options(options),
      _M_clone(accelerator()),
      _M_scale(options.scale)
    {
        _M_wq = register_polymorphic_layer<Linear>("wq");
        _M_wk = register_polymorphic_layer<Linear>("wk");
        _M_wv = register_polymorphic_layer<Linear>("wv");
        _M_wo = register_polymorphic_layer<Linear>("wo");

        caching_options cache_options{
            .head_dim = options.head_dim,
            .n_heads = options.n_heads,
            .n_kv_heads = options.n_kv_heads,
            .max_seq_len = options.max_seq_len,
            .max_batch_size = options.max_batch_size,
        };

        _M_cache = register_layer<Cache>("cache", cache_options);

        if (options.norm_eps) {
            enable_norm(options.norm_eps.value(), options.norm_mu.value_or(0.0f));
        }
    }

    attention(const attention_options& options, hardware_accelerator& accelerator)
    : attention(
          options,
          indirect_layer<RotaryPositionalEmbedding>(
              options.head_dim, options.max_seq_len, options.rope_theta, accelerator
          )
      )
    {}

    /// Enable RMS-normalization of keys and queries.
    void
    enable_norm(float eps, float mu)
    {
        _M_options.norm_eps = eps;
        _M_options.norm_mu = mu;
        _M_wq_norm = register_layer<RMSNorm>("q_norm", eps, mu);
        _M_wk_norm = register_layer<RMSNorm>("k_norm", eps, mu);
    }

    /// Compute multi-head attention of the input sequence.
    ///
    /// \param input an input embedding.
    /// \param mask if specified, a 2-dim mask preventing attention to certain positions.
    /// \param start_pos a start position of the input sequence.
    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, std::optional<Mask> mask = std::nullopt, std::size_t start_pos = 0)
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

        q = _M_rope(_M_wq_norm ? _M_wq_norm(q) : q, /*start_pos=*/start_pos);
        k = _M_rope(_M_wk_norm ? _M_wk_norm(k) : k, /*start_pos=*/start_pos);

        auto [kk, vv] = _M_cache->update(k, v, start_pos);

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

        auto scores = matmul(queries, keys, accelerator());
        scores = mul(scores, _M_scale, accelerator());
        if (mask.has_value()) {
            scores = add_broadcast(scores, mask.value(), accelerator());
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


/// This method performs adaptation of reference Meta-style K and Q weight matrices to the
/// HuggingFace-style attention weight matrices. This adaptation is necessary since MetalChat
/// uses HuggingFace's approach to compute rotary position embeddings.
///
/// \tparam Input type of the input tensor.
/// \param input input attention tensor.
/// \param n_heads number of the attention heads in the input tensor.
/// \param accelerator a reference to a hardware accelerator.
template <immutable_tensor2 Input> requires std::default_initializable<Input>
auto
permute_attention_heads(const Input& input, std::size_t n_heads, hardware_accelerator& accelerator)
{
    std::size_t size = input.sizes().front();
    std::size_t attention_heads = size / n_heads / 2;

    // Transposition of the dimension 1 and 2 results in a discontiguous container layout,
    // therefore we need to copy elements row-wise (by the last dimension).
    //
    // This implementation performs transposition on-the-fly, inserts rows into the necessary
    // positions given the strides of the input and output tensors.
    tensor_accessor input_layout({n_heads, attention_heads, 2});
    tensor_accessor output_layout({n_heads, 2, attention_heads});

    std::vector<Input> tensors(size);
    for (std::size_t input_index = 0; input_index < size; input_index++) {
        auto i = input_index / input_layout.stride(0);
        auto rem = input_index % input_layout.stride(0);
        auto j = rem / input_layout.stride(1);
        auto k = rem % input_layout.stride(1);

        const auto output_index = i * output_layout.stride(0) + k * output_layout.stride(1) + j;
        tensors[output_index] = input.narrow(0, input_index, 1);
    }

    using value_type = typename Input::value_type;
    auto output = concatenate<value_type>(tensors.begin(), tensors.end(), 0, accelerator);

    return output.get();
}


/// This method performs adaptation of HuggingFace-style attention weight matrices (K, Q) to
/// Meta-style shape.
///
/// \tparam T a data type of the input tensor.
/// \param ptr a pointer to the input tensor.
/// \param n_heads a number of the attention heads in the input tensor.
template <typename T>
void
permute_attention_heads(
    std::shared_ptr<basic_tensor>& ptr, std::size_t n_heads, hardware_accelerator& accelerator
)
{
    using container_type = hardware_memory_container<T>;
    using tensor_type = tensor<T, 2, container_type>;

    auto weight = shared_tensor(tensor_type());
    tensor_accessor::resize(*ptr, weight.accessor(), ptr->dimensions());
    weight.set_container(ptr->container_ptr());

    weight = permute_attention_heads(weight, n_heads, accelerator);
    ptr->set_container(weight.container_ptr());
}


template <typename T>
auto
make_causal_mask(std::size_t size0, std::size_t size1, hardware_accelerator& accelerator)
{
    auto alloc = accelerator.get_allocator();
    std::optional<future_tensor<T, 2>> mask;

    if (size0 > 1) {
        const T infinity = static_cast<T>(std::numeric_limits<float>::infinity());

        auto m = full<T>({size0, size1}, -infinity, alloc);
        triu(m.narrow(1, size1 - size0, size0), /*diagonal=*/1);
        mask = std::make_optional(std::move(m));
    }

    return mask;
}


template <typename T>
auto
make_sliding_causal_mask(
    std::size_t size0, std::size_t size1, std::size_t window, hardware_accelerator& accelerator
)
{
    auto alloc = accelerator.get_allocator();
    std::optional<future_tensor<T, 2>> mask;

    if (size0 > 1) {
        const T infinity = static_cast<T>(std::numeric_limits<float>::infinity());

        auto upper = full<T>({size0, size1}, -infinity, alloc);
        auto lower = full<T>({size0, size1}, -infinity, alloc);

        triu(upper.narrow(1, size1 - size0, size0), /*diagonal=*/1);
        triu(lower.narrow(1, size1 - size0, size0).t(), /*diagonal=*/window);

        auto m = add(upper, lower, accelerator);
        mask = std::make_optional(std::move(m));
    }

    return mask;
}


} // namespace nn
} // namespace metalchat

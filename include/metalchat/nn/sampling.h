// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>

#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace nn {


/// Provides access to sampling state consisting of raw logits and their
/// positions in the model vocabulary.
template <typename T, typename Index> struct basic_sampling_context {
    using value_type = T;
    using index_type = Index;
    using logits_tensor = future_tensor<value_type, 2>;
    using index_tensor = future_tensor<index_type, 2>;

    logits_tensor logits;
    index_tensor indices;
};


template <typename T> struct basic_sampler {
    using value_type = T;
    using index_type = int32_t;

    using context_type = basic_sampling_context<value_type, index_type>;
    using logits_tensor = context_type::logits_tensor;
    using index_tensor = context_type::index_tensor;

    template <immutable_tensor2_t<T> Tensor>
    static context_type
    construct_context(Tensor logits, hardware_accelerator& accelerator)
    {
        auto alloc = accelerator.get_allocator();
        auto indices = shared_empty_like<index_type>(logits, alloc);

        // TODO: replace with metal-accelerated arange kernel.
        for (std::size_t i = 0; i < indices.size(0); i++) {
            for (std::size_t j = 0; j < indices.size(1); j++) {
                indices[i][j] = j;
            }
        }

        return context_type{logits, future_tensor(indices)};
    }

    /// Return subset of raw logits and their indices (context) that should be
    /// considered in token sequence generation for a language transformer model.
    virtual context_type
    filter(const context_type& context, hardware_accelerator& accelerator) = 0;

    /// Return indices of the original logits that should be considered in token
    /// sequence generation for a language transformer model.
    virtual index_tensor
    sample(const context_type& context, hardware_accelerator& accelerator) = 0;

    template <immutable_tensor_t<T> Tensor>
    index_tensor
    sample(Tensor logits, hardware_accelerator& accelerator)
    {
        auto context = construct_context(logits, accelerator);
        context = filter(context, accelerator);
        return sample(context, accelerator);
    }

    /// A default virtual destructor.
    virtual ~basic_sampler() = default;
};


template <typename T, typename Sampler> struct basic_multinomial_sampler : public basic_sampler<T> {
public:
    using value_type = T;
    using base_type = basic_sampler<value_type>;
    using sampler_type = Sampler;
    using context_type = base_type::context_type;
    using logits_tensor = base_type::logits_tensor;
    using index_tensor = base_type::index_tensor;

    context_type
    filter(const context_type& context, hardware_accelerator& accelerator)
    {
        return static_cast<Sampler*>(this)->filter(context, accelerator);
    }

    index_tensor
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        auto next_token = multinomial(context.logits, /*sample_size=*/1, accelerator);
        return gather(context.indices, next_token, accelerator);
    }
};


/// A sampler that applies the provided samplers one after another, passing the output from
/// the previous sampler to the next one as an input.
template <typename T> class sequential_sampler : public basic_sampler<T> {
public:
    using value_type = T;
    using sampler_type = basic_sampler<T>;
    using sampler_pointer = std::shared_ptr<sampler_type>;
    using context_type = sampler_type::context_type;
    using logits_tensor = sampler_type::logits_tensor;
    using index_tensor = sampler_type::index_tensor;

    sequential_sampler(std::initializer_list<sampler_pointer> samplers)
    : _M_samplers(
          std::make_move_iterator(samplers.begin()), std::make_move_iterator(samplers.end())
      )
    {
        if (samplers.size() == 0) {
            throw std::invalid_argument("sequential_sampler: requires at least one sampler");
        }
    }

    context_type
    filter(const context_type& context, hardware_accelerator& accelerator)
    {
        auto sampling_context = context;
        for (auto& sampler : _M_samplers) {
            sampling_context = sampler->filter(sampling_context, accelerator);
        }
        return sampling_context;
    }

    index_tensor
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        return _M_samplers.back()->sample(context, accelerator);
    }

private:
    std::vector<sampler_pointer> _M_samplers;
};


template <typename T>
class nucleus_sampler : public basic_multinomial_sampler<T, nucleus_sampler<T>> {
private:
    T _M_temperature;
    T _M_p;

public:
    using value_type = T;
    using base_type = basic_sampler<T>;
    using context_type = base_type::context_type;
    using logits_tensor = base_type::logits_tensor;
    using index_tensor = base_type::index_tensor;

    nucleus_sampler(T temperature, T p)
    : _M_temperature(temperature),
      _M_p(p)
    {}

    nucleus_sampler()
    : nucleus_sampler(T(0.6), T(0.9))
    {}

    context_type
    filter(const context_type& context, hardware_accelerator& accelerator)
    {
        constexpr std::size_t BlockSize = 128;
        using Tensor = logits_tensor;

        T prob = T(1) - _M_p;
        T temp = T(1) / _M_temperature;

        auto logits = mul(context.logits, temp, accelerator);
        auto probs = softmax(logits, accelerator);

        auto [probs_sort, probs_idx] = sort(probs, accelerator);
        auto probs_sum = cumsum<Tensor, BlockSize>(probs_sort, accelerator);
        auto probs_diff = sub(probs_sum, probs_sort, accelerator);

        auto mask = gt(probs_diff, prob, accelerator);
        probs_sort = scatter(probs_sort, mask, T(0), accelerator);
        probs_idx = gather(context.indices, probs_idx, accelerator);

        return context_type{probs_sort, probs_idx};
    }
};


/// A CPU-based top-k logits sampling. It restricts the pool of candidate tokens to the k most
/// likely tokens.
///
/// The sampler processes each batch element independently, applying top-k filtering row-wise to
/// the input logits tensor. If k exceeds the vocabulary size, all tokens are retained.
///
/// \tparam T The data type of the logits (e.g., float, bf16)
template <typename T> class topk_sampler : public basic_sampler<T> {
private:
    std::size_t _M_k;

public:
    using value_type = T;
    using base_type = basic_sampler<T>;
    using context_type = base_type::context_type;
    using logits_tensor = base_type::logits_tensor;
    using index_tensor = base_type::index_tensor;

    /// Constructs a top-k sampler with the specified k value.
    ///
    /// \param k The number of top candidates to retain.
    ///          Keeps all tokens, if k is larger than vocabulary size.
    topk_sampler(std::size_t k)
    : _M_k(k)
    {}

    /// Applies top-k sampling to the input logits tensor.
    ///
    /// \param context input logits and indices of shape [batch_size, vocab_size].
    /// \param accelerator hardware accelerator used for tensor operations.
    context_type
    filter(const context_type& context, hardware_accelerator& accelerator)
    {
        using index_type = context_type::index_type;

        auto k = std::min(context.logits.size(1), _M_k);
        auto values = clone(context.logits, accelerator).get();
        auto indices = clone(context.indices, accelerator).get();

        for (std::size_t i = 0; i < indices.size(0); i++) {
            auto value = values[i].data_ptr();
            auto index = indices[i].data_ptr();
            auto cmp = [&](int32_t i1, int32_t i2) { return value[i1] > value[i2]; };
            std::partial_sort(index, index + k, index + indices.size(1), cmp);
        }

        indices = indices[slice(), slice(0, k)];
        auto logits = gather(values, indices, accelerator);

        return context_type{logits, indices};
    }

    index_tensor
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        auto probs = softmax(context.logits, accelerator);
        auto next_token = multinomial(probs, /*sample_size=*/1, accelerator);
        return gather(context.indices, next_token, accelerator);
    }
};


template <typename T>
std::shared_ptr<sequential_sampler<T>>
make_default_sampler()
{
    sequential_sampler<T> sampler(
        {std::make_shared<topk_sampler<T>>(50), std::make_shared<nucleus_sampler<T>>()}
    );

    return std::make_shared<sequential_sampler<T>>(std::move(sampler));
}


} // namespace nn
} // namespace metalchat

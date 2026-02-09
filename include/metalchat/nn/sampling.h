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
    /// The value type of logits tensor.
    using value_type = T;
    /// The value type of index tensor.
    using index_type = Index;
    /// Logits tensor type.
    using logits_tensor = future_tensor<value_type, 2>;
    /// Index tensor type.
    using index_tensor = future_tensor<index_type, 2>;

    /// The logits.
    logits_tensor logits;
    /// The logits indices.
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
    sample(const context_type& context, hardware_accelerator& accelerator) = 0;

    template <immutable_tensor_t<T> Tensor>
    index_tensor
    sample(Tensor logits, hardware_accelerator& accelerator)
    {
        auto context = construct_context(logits, accelerator);
        context = sample(context, accelerator);
        return context.indices;
    }

    /// A default virtual destructor.
    virtual ~basic_sampler() = default;
};


/// A sampler that applies the provided samplers one after another, passing the output from
/// the previous sampler to the next one as an input.
///
/// Here is an example how to create the most common sampling strategy using a composition
/// of \ref nucleus_sampler and \ref multinomial_sampler.
/// ```c++
/// using namespace metalchat::nn;
///
/// auto sampler = sequential_sampler<float>({
///     std::make_shared<nucleus_sampler<float>>(),
///     std::make_shared<multinomial_sampler<float>>()
/// });
/// ```
///
/// \note When sequential sampler is created from an empty range, the sequential sampler
/// returns unmodified sampling context.
template <typename T> class sequential_sampler : public basic_sampler<T> {
public:
    /// A base type for samplers comprising a \ref sequential_sampler.
    using sampler_type = basic_sampler<T>;
    /// A shared pointer type to the \ref sampler_type.
    using sampler_pointer = std::shared_ptr<sampler_type>;

    using value_type = T;
    using index_type = int32_t;

    using context_type = basic_sampling_context<value_type, index_type>;
    using logits_tensor = context_type::logits_tensor;
    using index_tensor = context_type::index_tensor;

    /// Constructs the \ref sequential_sampler from the list of samplers.
    sequential_sampler(std::initializer_list<sampler_pointer> samplers)
    : sequential_sampler(samplers.begin(), samplers.end())
    {}

    /// Constructs the \ref sequential_sampler by moving elements from the specified range.
    ///
    /// \param first, last the pair of iterators defining the range of elements to move
    /// the samplers from.
    template <std::forward_iterator ForwardIt>
    sequential_sampler(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, sampler_pointer>
    : _M_samplers(std::make_move_iterator(first), std::make_move_iterator(last))
    {}

    /// The default \ref sequential_sampler constructor.
    sequential_sampler()
    : _M_samplers({})
    {}

    context_type
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        auto sampling_context = context;
        for (auto& sampler : _M_samplers) {
            sampling_context = sampler->sample(sampling_context, accelerator);
        }
        return sampling_context;
    }

private:
    std::vector<sampler_pointer> _M_samplers;
};


/// A sampler that selects the smallest set of elements whose cumulative probability
/// exceeds the probability `p`.
///
/// This version of a sampler combines top-p sampling with temperature scaling.
template <typename T> class nucleus_sampler : public basic_sampler<T> {
public:
    using value_type = T;
    using index_type = int32_t;

    using context_type = basic_sampling_context<value_type, index_type>;
    using logits_tensor = context_type::logits_tensor;
    using index_tensor = context_type::index_tensor;

    /// The \ref nucleus_sampler constructor.
    ///
    /// \param temperature a positive value used to modulate the logits distribution.
    /// \param p the cumulative probability cutoff value.
    nucleus_sampler(T temperature, T p)
    : _M_temperature(temperature),
      _M_p(p)
    {
        if (temperature <= T(0)) {
            throw std::invalid_argument("nucleus_sampler: temperature must be positive");
        }
        if (p < T(0) || p > T(1.0)) {
            throw std::invalid_argument("nucleus_sampler: probability must be in [0.0, 1.0]");
        }
    }

    /// The default \ref nucleus_sampler constructor initializers `temperature`
    /// parameters with `0.6`, and `p` parameter with `0.9` value.
    nucleus_sampler()
    : nucleus_sampler(T(0.6), T(0.9))
    {}

    context_type
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        T temp = T(1) / _M_temperature;

        auto logits = mul(context.logits, temp, accelerator);
        auto probs = softmax(logits, accelerator);

        auto [probs_sort, probs_idx] = sort(probs, accelerator);
        auto probs_sum = cumsum(probs_sort, accelerator);
        auto probs_diff = sub(probs_sum, probs_sort, accelerator);

        auto mask = gt(probs_diff, _M_p, accelerator);
        probs_sort = scatter(probs_sort, mask, T(0), accelerator);
        probs_idx = gather(context.indices, probs_idx, accelerator);

        return context_type{probs_sort, probs_idx};
    }

private:
    T _M_temperature;
    T _M_p;
};


/// A CPU-based top-k logits sampling. It restricts the pool of candidate tokens to the k most
/// likely tokens.
///
/// The sampler processes each batch element independently, applying top-k filtering row-wise to
/// the input logits tensor. If k exceeds the vocabulary size, all tokens are retained.
///
/// \warning This implementation uses a CPU-based
/// [selection algorithm](https://en.wikipedia.org/wiki/Selection_algorithm) to find top-k
/// largest elements in the logits tensor. This implies that pending command queue is submitted
/// to GPU for processing and the result (logits) are awaited by blocking a thread.
///
/// \tparam T The data type of the logits (e.g., float, bf16)
template <typename T> class topk_sampler : public basic_sampler<T> {
private:
    std::size_t _M_k;

public:
    using value_type = T;
    using index_type = int32_t;

    using context_type = basic_sampling_context<value_type, index_type>;
    using logits_tensor = context_type::logits_tensor;
    using index_tensor = context_type::index_tensor;

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
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        using index_type = context_type::index_type;

        auto k = std::min(context.logits.size(1), _M_k);
        auto values = clone(context.logits, accelerator).get();
        auto indices = clone(context.indices, accelerator).get();

        for (std::size_t i = 0; i < indices.size(0); i++) {
            auto value = values[i].data_ptr();
            auto index = indices[i].data_ptr();
            auto cmp = [&](index_type i1, index_type i2) { return value[i1] > value[i2]; };
            std::partial_sort(index, index + k, index + indices.size(1), cmp);
        }

        indices = indices[slice(), slice(0, k)];
        auto logits = gather(values, indices, accelerator);

        return context_type{logits, indices};
    }
};


/// Draws samples interpreting logits array as a cumulative distribution function
/// of a multinomial distribution.
///
/// \warning In order to using this sampler, the logits must be sorted in a descending
/// order, otherwise the result is undefined.
template <typename T> class multinomial_sampler : public basic_sampler<T> {
public:
    using value_type = T;
    using index_type = int32_t;

    using context_type = basic_sampling_context<value_type, index_type>;
    using logits_tensor = context_type::logits_tensor;
    using index_tensor = context_type::index_tensor;

    /// The \ref multinomial_sampler constructor.
    ///
    /// \param sample_size a number of samples that should be drawn from the distribution.
    multinomial_sampler(std::size_t sample_size = 1)
    : _M_sample_size(sample_size)
    {}

    context_type
    sample(const context_type& context, hardware_accelerator& accelerator)
    {
        auto next_token = multinomial(context.logits, _M_sample_size, accelerator);
        auto logits = gather(context.logits, next_token, accelerator);
        auto indices = gather(context.indices, next_token, accelerator);

        return context_type{logits, indices};
    }

private:
    std::size_t _M_sample_size;
};


template <typename T>
std::shared_ptr<sequential_sampler<T>>
make_default_sampler(std::size_t sample_size = 1)
{
    sequential_sampler<T> sampler(
        {std::make_shared<topk_sampler<T>>(std::max(sample_size, std::size_t(50))),
         std::make_shared<nucleus_sampler<T>>(),
         std::make_shared<multinomial_sampler<T>>(sample_size)}
    );

    return std::make_shared<sequential_sampler<T>>(std::move(sampler));
}


} // namespace nn
} // namespace metalchat

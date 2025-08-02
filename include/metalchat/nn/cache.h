#pragma once

#include <bit>
#include <vector>

#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {
namespace nn {


template <typename Cache>
using __cache_update_r = std::
    tuple<typename Cache::input_tensor, typename Cache::input_tensor, typename Cache::mask_tensor>;


struct cache_options {
    std::size_t head_dim;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t max_seq_len;
    std::size_t max_batch_size;
};


template <typename Cache>
concept cache = requires(Cache cache) {
    typename Cache::value_type;

    /// The first element of the `update` return value should be a cached
    /// data (with the same dimensionality as an input tensor).
    typename Cache::input_tensor;
    requires immutable_tensor_t<typename Cache::input_tensor, typename Cache::value_type>;

    /// The third element of the `update` return value should be a causal
    /// additive mask. When the size of the input query is different from 1,
    /// then causal mask will be non-empty.
    typename Cache::mask_tensor;
    requires optional_tensor_t<typename Cache::mask_tensor, typename Cache::value_type>;

    {
        cache.update(
            std::declval<typename Cache::input_tensor>(),
            std::declval<typename Cache::input_tensor>(), std::size_t()
        )
    } -> std::same_as<__cache_update_r<Cache>>;
};


template <typename CacheConstructible>
concept cache_constructible = requires {
    requires cache<CacheConstructible>;

    requires std::constructible_from<CacheConstructible, cache_options, hardware_accelerator>;
};


template <typename Cache, typename T>
concept cache_t = cache<Cache> && std::same_as<typename Cache::value_type, T>;


template <typename T> class sink_cache {
public:
    using value_type = T;
    using input_tensor = future_tensor<T, 4>;
    using mask_tensor = std::optional<future_tensor<T, 2>>;
    using return_type = std::tuple<input_tensor, input_tensor, mask_tensor>;

    sink_cache(std::size_t pre_len, const cache_options& options, hardware_accelerator accelerator)
    : _M_accelerator(accelerator),
      _M_copy(accelerator),
      _M_options(options),
      _M_keys(alloc(options)),
      _M_vals(alloc(options)),
      _M_pre_len(pre_len)
    {}

    sink_cache(const cache_options& options, hardware_accelerator accelerator)
    : sink_cache(std::bit_width(options.max_seq_len) - 1, options, accelerator)
    {}

    return_type
    update(input_tensor keys, input_tensor vals, std::size_t start_pos)
    {
        if (keys.size(1) != vals.size(1)) {
            throw std::invalid_argument(std::format(
                "sink_cache: length of key is different from vals ({} != {})", keys.size(1),
                vals.size(1)
            ));
        }

        auto [cache_keys, k] = copy(_M_keys, keys, start_pos);
        auto [cache_vals, v] = copy(_M_vals, vals, start_pos);

        _M_keys = cache_keys;
        _M_vals = cache_vals;

        auto len = keys.size(1);
        auto mask_len = k.size(1);
        auto mask = create_additive_causal_mask(len, mask_len);

        return std::make_tuple(k, v, mask);
    }

private:
    auto
    create_additive_causal_mask(std::size_t len, std::size_t mask_len) const
    {
        mask_tensor mask;

        if (len > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());

            auto alloc = _M_accelerator.get_allocator();
            auto m = full<T>({len, mask_len}, -infinity, alloc);

            triu(m.narrow(1, mask_len - len, len));
            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

    auto
    alloc(const cache_options& options)
    {
        return future_tensor(empty<T>(
            {options.max_batch_size, options.max_seq_len, options.n_kv_heads, options.head_dim},
            _M_accelerator.get_allocator()
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
    template <immutable_hardware_tensor4_t<T> Cache, immutable_tensor4_t<T> Input>
    auto
    copy(Cache cache, Input input, std::size_t start_pos)
    {
        using s = indexing::slice;
        const auto bs = input.size(0);
        const auto len = input.size(1);
        const auto cache_size = cache.size(1);

        const std::size_t post_len = cache_size - _M_pre_len;

        if (len > cache_size) {
            throw std::invalid_argument(std::format(
                "sink_cache: requested length ({}) is larger than the cache size ({})", len,
                cache_size
            ));
        }

        // When the cache is full (meaning that start position spilled over the boundaries
        // of the cache), rotate it left and store the inferred results into the right-most
        // position.
        if (start_pos >= cache_size) {
            auto cache_new = alloc(_M_options);
            auto cache_new_pre = cache_new.narrow(1, 0, _M_pre_len);
            auto cache_pre = cache.narrow(1, 0, _M_pre_len);

            // Copy the prefix of the cache to a newly allocated memory.
            cache_new = future_tensor(cache_new, _M_copy(cache_pre, cache_new_pre));

            auto cache_new_post = cache_new.narrow(1, _M_pre_len, post_len);
            auto cache_post = cache.narrow(1, _M_pre_len, post_len);

            // Roll the remaining cache by the size of a new input length.
            auto cache_rolled = roll(cache_post, cache_new_post, int32_t(len), 1, _M_accelerator);

            cache = future_tensor(cache_new, cache_rolled);

            start_pos = cache_size - len;
        }

        // Write the result of computation into the "target" tensor, so we could reuse
        // it on the next iteration again. To make precise inference, model will use all
        // previously cached results (or up to the end position).
        auto end_pos = start_pos + len;
        auto target = cache[s(0, bs), s(start_pos, end_pos), s(), s()];

        cache = future_tensor(cache, _M_copy(input, target));
        auto cached_data = cache[s(0, bs), s(0, end_pos), s(), s()];

        return std::make_tuple(cache, cached_data);
    }

    hardware_accelerator _M_accelerator;
    kernel::cpy<value_type> _M_copy;
    cache_options _M_options;

    input_tensor _M_keys;
    input_tensor _M_vals;

    std::size_t _M_pre_len;
};


} // namespace nn
} // namespace metalchat

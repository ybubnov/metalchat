#pragma once

#include <bit>
#include <vector>

#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/nn/layer.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {
namespace nn {


/// The result of a cache update query. The dimensions of the \ref caching_result::keys and
/// \ref caching_result::values tensors are like following: [`bs`, `max_seq_len`, `n_kv_heads`,
/// `n_heads`].
///
/// \tparam T data type of the cache elements (float, int, etc.).
template <typename T> struct caching_result {
    /// A future tensor that will contain a result of keys caching query.
    future_tensor<T, 4> keys;

    /// A future tensor that will contain a result of values caching query.
    future_tensor<T, 4> values;

    /// An optional future tensor that will contain a result of additive causal mask creation.
    /// This mask is created only when the length of keys (or values) is larger than `1`.
    std::optional<future_tensor<T, 2>> mask;
};


/// Caching options to configure the key-value cache of the large language model.
struct caching_options {
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
};


template <typename Cache>
concept cache = requires(Cache cache) {
    typename Cache::value_type;

    /// The first element of the `update` return value should be a cached
    /// data (with the same dimensionality as an input tensor).
    typename Cache::input_tensor;
    requires immutable_tensor_t<typename Cache::input_tensor, typename Cache::value_type>;

    {
        cache.update(
            std::declval<typename Cache::input_tensor>(),
            std::declval<typename Cache::input_tensor>(), std::size_t()
        )
    } -> std::same_as<caching_result<typename Cache::value_type>>;
};


template <typename CacheConstructible>
concept cache_constructible = requires {
    requires cache<CacheConstructible>;

    requires std::constructible_from<CacheConstructible, caching_options, hardware_accelerator>;
};


template <typename Cache, typename T>
concept cache_t = cache<Cache> && std::same_as<typename Cache::value_type, T>;


/// Implementation of the KV cache introduced in attention sinks paper. It allows the model to
/// generate text beyond the length of its context window, without losing fluency in the
/// conversation.
///
/// This is done by always keeping the first few tokens ("sink tokens") in the KV cache, as
/// models often pay a large amount of attention to them during the training. As it discards
/// past non-sink tokens, the model will lose the ability to generate tokens that depend on
/// the context that was discarded. It's also a solution to contain the memory footprint of
/// the KV cache.
///
/// \warning The implementation does not track if the specified start position corresponds
/// to the latest used start position. So if a user calls an attention layer with
/// `start_pos = 15` and cache size set to `16`, and then makes subsequent call with
/// `start_pos = 44`, the implementation won't complain, but the result might be not what a
/// user expects.
///
/// \tparam T data type of the cache elements (float, int, etc.).
template <typename T> class sink_cache : public basic_layer {
public:
    using value_type = T;
    using input_tensor = future_tensor<T, 4>;
    using mask_tensor = std::optional<future_tensor<T, 2>>;

    /// Constructs a new instance of the sink cache.
    ///
    /// \param pre_len a number of sink tokens that will be permanently kept in cache.
    /// \param options caching options for the KV cache.
    /// \param accelerator a hardware accelerator instance.
    sink_cache(
        std::size_t pre_len, const caching_options& options, hardware_accelerator accelerator
    )
    : basic_layer(accelerator),
      _M_clone(accelerator),
      _M_options(options),
      _M_keys(alloc(options)),
      _M_vals(alloc(options)),
      _M_pre_len(pre_len)
    {
        update_parameters();
    }

    /// Constructs a new instance of the sink cache with the number of sink tokens set to the
    /// logarithm of base 2 from the maximum length of the context window.
    ///
    /// \param options caching options for the KV cache.
    /// \param accelerator a hardware accelerator instance.
    sink_cache(const caching_options& options, hardware_accelerator accelerator)
    : sink_cache(std::bit_width(options.max_seq_len) - 1, options, accelerator)
    {}

    /// Updates the cache tensor with new keys and values.
    ///
    /// \param keys new keys to cache.
    /// \param vals new values to cache.
    /// \param start_pos position of the next token in an output sequence.
    caching_result<T>
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
        update_parameters();

        auto len = keys.size(1);
        auto mask_len = k.size(1);
        auto mask = create_additive_causal_mask(len, mask_len);

        return caching_result{k, v, mask};
    }

private:
    auto
    create_additive_causal_mask(std::size_t len, std::size_t mask_len) const
    {
        mask_tensor mask;

        if (len > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());

            auto alloc = accelerator().get_allocator();
            auto m = full<T>({len, mask_len}, -infinity, alloc);

            triu(m.narrow(1, mask_len - len, len));
            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

    auto
    alloc(const caching_options& options)
    {
        return future_tensor(empty<T>(
            {options.max_batch_size, options.max_seq_len, options.n_kv_heads, options.head_dim},
            accelerator().get_allocator()
        ));
    }

    /// Cache the intermediate results (Rotation Positional Encodings) into the cache tensor.
    ///
    /// The implementation allows to store the inference results for the position larger than
    /// the cache size: it simply drops the least recent results. It works like a sliding window.
    template <immutable_hardware_tensor4_t<T> Cache, immutable_tensor4_t<T> Input>
    auto
    copy(Cache cache, Input input, std::size_t start_pos)
    {
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
            cache_new = future_tensor(cache_new, _M_clone(cache_pre, cache_new_pre));

            auto cache_new_post = cache_new.narrow(1, _M_pre_len, post_len);
            auto cache_post = cache.narrow(1, _M_pre_len, post_len);

            // Roll the remaining cache by the size of a new input length.
            auto cache_rolled = roll(cache_post, cache_new_post, int32_t(len), 1, accelerator());

            cache = future_tensor(cache_new, cache_rolled);

            start_pos = cache_size - len;
        }

        // Write the result of computation into the "target" tensor, so we could reuse
        // it on the next iteration again. To make precise inference, model will use all
        // previously cached results (or up to the end position).
        auto end_pos = start_pos + len;
        auto target = cache[slice(0, bs), slice(start_pos, end_pos), slice(), slice()];

        cache = future_tensor(cache, _M_clone(input, target));
        auto cached_data = cache[slice(0, bs), slice(0, end_pos), slice(), slice()];

        return std::make_tuple(cache, cached_data);
    }

    void
    update_parameters()
    {
        register_parameter("keys", _M_keys.get_nowait());
        register_parameter("values", _M_vals.get_nowait());
    }

    kernel::clone<value_type> _M_clone;
    caching_options _M_options;

    input_tensor _M_keys;
    input_tensor _M_vals;

    std::size_t _M_pre_len;
};


} // namespace nn
} // namespace metalchat

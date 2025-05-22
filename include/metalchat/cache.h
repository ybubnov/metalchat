#pragma once

#include <metalchat/tensor/concept.h>


namespace metalchat {


template <typename Input, typename Cache>
using __cache_update_r = decltype(cache.update(std::declval<const Input>()));


template <std::size_t I, typename Input, typename Cache>
using __cache_update_element_t = std::tuple_element_t<I, __cache_update_r<Input, Cache>>;


template <typename Input, typename Cache>
concept __cache = requires(Cache cache, const Input input) {
    { cache.update(input) } -> std::same_as<__cache_update_r<Input, Cache>>;

    requires immutable_tensor<Input>;

    /// The first element of the tuple's return value should be a cached
    /// data (with the same dimensionality as an input tensor).
    requires immutable_tensor<__cache_update_element_t<0, Input, Cache>>;

    /// The second element of the tuple's return value should be a causal
    /// additive mask. When the size of the input query is different from 1,
    /// then causal mask will be non-empty.
    requires optional_tensor_v<__cache_update_element_t<1, Input, Cache>>;
};


template <typename Cache, typename T>
concept cache = __cache<future_tensor<T, 4>, Cache>;


} // namespace metalchat

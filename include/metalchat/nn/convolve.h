// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/functional.h>
#include <metalchat/kernel/convolve.h>
#include <metalchat/nn/cache.h>
#include <metalchat/nn/layer.h>
#include <metalchat/nn/linear.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {
namespace nn {


/// Applies a 1D convolution over an input signal composed of several input planes.
template <typename T, contiguous_container Container = hardware_memory_container<T>>
class conv1d : public basic_layer {
public:
    using value_type = T;
    using container_type = Container;
    using weight_type = tensor<T, 3, Container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    /// The 1-dimensional convolution layer constructor.
    ///
    /// \param in_channels The number of channels in the input.
    /// \param out_channels The number of channels produced by convolution.
    /// \param kernel_size The size of the convolution kernel.
    /// \param padding Zero-padding added to both sides of the input.
    /// \param groups The number of block-connections from input channels to output channels.
    /// \param accelerator The hardware accelerator.
    conv1d(
        std::size_t in_channels,
        std::size_t out_channels,
        std::size_t kernel_size,
        std::size_t padding,
        std::size_t groups,
        hardware_accelerator& accelerator
    ) requires std::same_as<Container, hardware_memory_container<T>>
    : basic_layer(accelerator),
      _M_kernel(accelerator),
      _M_weight(nullptr),
      _M_padding(padding),
      _M_groups(groups)
    {
        if (in_channels % groups) {
            throw std::invalid_argument("nn::conv1d: in_channels must be divisible by groups");
        }

        auto alloc = rebind_allocator<T, Allocator>(accelerator.get_allocator());
        auto weight = rand<T>({out_channels, in_channels / groups, kernel_size}, alloc);

        _M_weight = shared_tensor(std::move(weight));
        initialize();
    }

    conv1d(std::size_t padding, std::size_t groups, hardware_accelerator& accelerator)
    : conv1d(shared_tensor(weight_type()), padding, groups, accelerator)
    {}

    template <immutable_tensor3_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_conv1d(input, _M_weight, padding, groups);
    }

    template <immutable_tensor2_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_conv1d(input, _M_weight, padding, groups);
    }

private:
    conv1d(
        weight_pointer weight,
        std::size_t padding,
        std::size_t groups,
        hardware_accelerator& acelerator
    )
    : basic_layer(accelerator),
      _M_conv(accelerator),
      _M_weight(weight),
      _M_padding(padding),
      _M_groups(groups)
    {
        initialize();
    }

    void
    initialize()
    {
        register_parameter("weight", _M_weight);
    }

    kernel::conv1d<T> _M_conv;
    weight_pointer _M_weight;
    std::size_t _M_padding;
    std::size_t _M_groups;
};


template <typename T, contiguous_container Container, mutable_layer Cache = window_cache<T>>
class short_conv : public basic_layer {
private:
    using Linear = linear<T, Container>;
    using Conv1d = conv1d<T, Container>;

public:
    using value_type = T;
    using container_type = Container;

    short_conv(std::size_t groups, hardware_accelerator& accelerator)
    : basic_layer(accelerator)
    {
        _M_in_proj = register_layer<Linear>("in_proj");
        _M_out_proj = register_layer<Linear>("out_proj");
        _M_conv = register_layer<Conv1d>("conv", /*padding=*/0, /*groups=*/groups);
        _M_cache = register_layer<Cache>("cache");
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Mask>
    auto
    operator()(Input input, std::optional<Mask> mask = std::nullopt, std::size_t start_pos = 0)
    {
        auto len = input.size(2);

        auto BCx = _M_in_proj(input).transpose({0, 2, 1});
        auto [B, C, x] = chunk(BCx, 3, /*dim=*/1);

        auto hidden = hadamard(B, x, accelerator());
        hidden = _M_cache.update(hidden, start_pos);
        hidden = _M_conv(hidden);

        // After the cache update, input will contain a padding containing the cached
        // rolling window. Drop the convolution of that window leaving only a tensor
        // of the input length.
        hidden = hidden.narrow(2, hidden.size(2) - len, len);

        hidden = hadamard(hidden, C, accelerator()).transpose({0, 2, 1});
        return _M_out_proj(hidden);
    }

private:
    indirect_layer<Linear> _M_in_proj;
    indirect_layer<Linear> _M_out_proj;
    indirect_layer<Conv1d> _M_conv;
    indirect_layer<Cache> _M_cache;
};


} // namespace nn
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/functional.h>
#include <metalchat/kernel/convolve.h>
#include <metalchat/nn/layer.h>
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
      _M_conv(accelerator),
      _M_weight(nullptr),
      _M_padding(padding),
      _M_groups(groups)
    {
        if (in_channels % groups) {
            throw std::invalid_argument("nn::conv1d: in_channels must be divisible by groups");
        }

        auto alloc = accelerator.get_allocator();
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
        return _M_conv(input, _M_weight, _M_padding, _M_groups);
    }

    template <immutable_tensor2_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_conv(input, _M_weight, _M_padding, _M_groups);
    }

private:
    conv1d(
        weight_pointer weight,
        std::size_t padding,
        std::size_t groups,
        hardware_accelerator& accelerator
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


} // namespace nn
} // namespace metalchat

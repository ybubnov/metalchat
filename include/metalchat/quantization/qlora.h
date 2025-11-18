// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/container.h>
#include <metalchat/nn/linear.h>


namespace metalchat {
namespace quantization {


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class qlora_adaptor : public nn::basic_layer {
private:
    using _Linear = nn::linear<T, Container>;

    _Linear::layer_pointer _M_a;
    _Linear::layer_pointer _M_b;

public:
    using value_type = T;
    using container_type = Container;

    using layer_type = qlora_adaptor<T, Container>;
    using layer_pointer = nn::shared_layer_ptr<layer_type>;

    qlora_adaptor(
        std::size_t in_features,
        std::size_t out_features,
        std::size_t rank,
        hardware_accelerator& accelerator
    )
    : basic_layer(accelerator)
    {
        _M_a = register_layer("A", _Linear(in_features, rank, accelerator));
        _M_b = register_layer("B", _Linear(rank, out_features, accelerator));
    }

    qlora_adaptor(hardware_accelerator& accelerator)
    : basic_layer(accelerator)
    {
        _M_a = register_layer("A", _Linear(accelerator));
        _M_b = register_layer("B", _Linear(accelerator));
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_b(_M_a(input));
    }
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class qlora_linear : public nn::basic_linear<T, Container> {
private:
    using _Base = nn::basic_linear<T, Container>;
    using _Adaptor = qlora_adaptor<T, Container>;
    using _Container = container_remove_type<Container>::type;

    using weight_container = container_rebind<std::int8_t, _Container>::type;
    using weight_type = tensor<std::int8_t, 2, weight_container>;
    using weight_pointer = shared_tensor_ptr<weight_type>;

    using scales_container = container_rebind<float, _Container>::type;
    using scales_type = tensor<float, 2, scales_container>;
    using scales_pointer = shared_tensor_ptr<scales_type>;

    _Adaptor::layer_pointer _M_adaptor;

    weight_pointer _M_weight;
    scales_pointer _M_scales;

public:
    using value_type = T;
    using container_type = Container;

    qlora_linear(float scale, hardware_accelerator& accelerator)
    : _Base(accelerator),
      _M_adaptor(nullptr),
      _M_weight(weight_type()),
      _M_scales(scales_type())
    {
        _M_adaptor = _Base::register_layer("adaptor", _Adaptor(accelerator));
        _Base::register_parameter("weight", _M_weight);
        _Base::register_parameter("scales", _M_scales);
    }

    _Base::result_type
    operator()(_Base::input_type input)
    {
        throw std::runtime_error("not implemented");
    }
};


} // namespace quantization
} // namespace metalchat

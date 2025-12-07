// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/container.h>
#include <metalchat/nn/linear.h>
#include <metalchat/tensor.h>


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
public:
    using value_type = T;
    using container_type = Container;

private:
    using _Base = nn::basic_linear<T, Container>;
    using _Adaptor = qlora_adaptor<T, Container>;

    using weight_traits = tensor_traits<std::int8_t, 2, container_type>;
    using scales_traits = tensor_traits<float, 2, container_type>;

    _Adaptor::layer_pointer _M_adaptor;

    std::size_t _M_group_size;
    weight_traits::pointer _M_weight;
    scales_traits::pointer _M_scales;
    value_type _M_scale;

public:
    qlora_linear(T scale, std::size_t group_size, hardware_accelerator& accelerator)
    : _Base(accelerator),
      _M_adaptor(nullptr),
      _M_group_size(group_size),
      _M_weight(typename weight_traits::type()),
      _M_scales(typename scales_traits::type()),
      _M_scale(scale)
    {
        _M_adaptor = _Base::register_layer("adaptor", _Adaptor(accelerator));
        _Base::register_parameter("weight", _M_weight);
        _Base::register_parameter("scales", _M_scales);
    }

    _Base::result_type
    operator()(_Base::input_type input)
    {
        auto& accelerator = _Base::accelerator();

        // The parameters in this module are defined as 2-dimensional tensors, since
        // original model is distributed with 2-dimensional weights. So in order to
        // load this module from a file, the reshaping must be implemented on-the-fly.
        //
        // Note: obviously model could be distributed with 3-dimensional tensors, so
        // this operation could be avoided. But seems like it's a historical artifact.
        auto weight_size = static_cast<int>(_M_weight.size(0));
        auto scales_size = static_cast<int>(_M_scales.size(0));
        auto group_size = static_cast<int>(_M_group_size);

        auto weight_lora = _M_weight.view({weight_size, -1, group_size});
        auto scales_lora = _M_scales.view({scales_size, -1, 1});

        // Note: hadamard multiplication stores the result into target type T (bfloat16,
        // by default). And the next matrix multiplication operation is performed in that
        // target type T (bfloat16). Which might result in a loss of precision.
        auto weight_dequant = hadamard_broadcast<T>(weight_lora, scales_lora, accelerator);
        auto weight = weight_dequant.view(_M_weight.shape());
        auto output = matmul(input, weight.transpose({1, 0}), accelerator);

        auto adaptation = mul(_M_adaptor(input), _M_scale, accelerator);
        return add(output, adaptation, accelerator);
    }
};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class qlora_embedding : public nn::basic_embedding<T, Container> {
private:
    using _Base = nn::basic_embedding<T, Container>;

    using weight_traits = tensor_traits<std::int8_t, 2, Container>;
    using scales_traits = tensor_traits<float, 2, Container>;

    weight_traits::pointer _M_weight;
    scales_traits::pointer _M_scales;
    kernel::embedding<T> _M_embedding;

public:
    using value_type = T;
    using container_type = Container;

    qlora_embedding(hardware_accelerator& accelerator)
    : _Base(accelerator),
      _M_weight(typename weight_traits::type()),
      _M_scales(typename scales_traits::type()),
      _M_embedding(accelerator)
    {
        _Base::register_parameter("weight", _M_weight);
        _Base::register_parameter("scales", _M_scales);
    }

    _Base::result_type
    operator()(_Base::input_type input)
    {
        std::cout << "qlora embedding" << std::endl;
        auto& accelerator = _Base::accelerator();
        auto weight_dequant = hadamard_broadcast<T>(_M_weight, _M_scales, accelerator);
        return _M_embedding(input, weight_dequant);
    }
};


} // namespace quantization
} // namespace metalchat

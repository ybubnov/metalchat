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
class linear : public nn::basic_linear<T, Container> {
private:
    using _Base = nn::basic_linear<T, Container>;

    using weight_traits = tensor_traits<std::int8_t, 2, Container>;
    using scales_traits = tensor_traits<float, 2, Container>;

    weight_traits::pointer _M_weight;
    scales_traits::pointer _M_scales;

    future_tensor<T, 2> _M_weight_dequant;
    bool _M_weight_done;

public:
    using value_type = T;
    using container_type = Container;

    linear(hardware_accelerator& accelerator)
    : _Base(accelerator),
      _M_weight(typename weight_traits::type()),
      _M_scales(typename scales_traits::type()),
      _M_weight_dequant(),
      _M_weight_done(false)
    {
        _Base::register_parameter("weight", _M_weight);
        _Base::register_parameter("scales", _M_scales);
    }

    _Base::result_type
    operator()(_Base::input_type input)
    {
        auto& accelerator = _Base::accelerator();

        if (!_M_weight_done) {
            _M_weight_dequant = hadamard_broadcast<T>(_M_weight, _M_scales, accelerator);
            _M_weight_done = true;
        }
        // return matmul(input, weight_dequant.transpose({1, 0}), accelerator);
        return matmul(input, _M_weight_dequant.transpose({1, 0}), accelerator);
    }
};


} // namespace quantization
} // namespace metalchat

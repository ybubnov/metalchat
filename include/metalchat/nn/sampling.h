// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace nn {


template <typename T> class basic_sampler {
public:
    using value_type = T;
    using index_type = int32_t;
    using input_tensor = future_tensor<value_type, 2>;
    using output_tensor = future_tensor<index_type, 2>;

    virtual output_tensor
    sample(input_tensor logits, hardware_accelerator& accelerator) = 0;

    template <immutable_tensor_t<T> InputTensor>
    output_tensor
    sample(InputTensor logits, hardware_accelerator& accelerator)
    {
        return sample(future_tensor(logits, accelerator));
    }

    virtual ~basic_sampler() {}
};


template <typename T> class nucleus_sampler : public basic_sampler<T> {
private:
    using _Base = basic_sampler<T>;

public:
    nucleus_sampler(T temperature, T p)
    : _M_temperature(temperature),
      _M_p(p)
    {}

    nucleus_sampler()
    : nucleus_sampler(T(0.6), T(0.9))
    {}

    _Base::output_tensor
    sample(_Base::input_tensor logits, hardware_accelerator& accelerator)
    {
        return top_p(logits, _M_temperature, _M_p, accelerator);
    }

private:
    T _M_temperature;
    T _M_p;
};


} // namespace nn
} // namespace metalchat

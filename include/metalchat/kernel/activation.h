// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


/// Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
template <typename T> class silu {
private:
    unary_kernel_wrapper<T> _M_kernel;

public:
    silu(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("silu"))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_kernel(input);
    }
};


/// Applies the Gaussian Error Linear Units function.
template <typename T> class gelu {
private:
    unary_kernel_wrapper<T> _M_kernel;

public:
    gelu(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("gelu"))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return _M_kernel(input);
    }
};


} // namespace kernel
} // namespace metalchat

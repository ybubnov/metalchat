#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16> class hadamard {
private:
    binary_kernel_wrapper<T, BlockSize> _M_kernel;

public:
    hadamard(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("hadamard"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return _M_kernel(input1, input2);
    }
};


template <typename T, std::size_t BlockSize = 1> class scalar_mul {
private:
    binary_kernel_wrapper<T, BlockSize> _M_kernel;

public:
    scalar_mul(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("scalar_mul"))
    {}

    template <immutable_tensor_t<T> Input, immutable_scalar_t<T> Multiplier>
    auto
    operator()(Input input, Multiplier multiplier)
    {
        return _M_kernel(input, multiplier);
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, const T multiplier)
    {
        return _M_kernel(input, multiplier);
    }
};


} // namespace kernel
} // namespace metalchat

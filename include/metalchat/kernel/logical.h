#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 32> class gt {
private:
    binary_kernel_wrapper<T, BlockSize> _M_kernel;

public:
    gt(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("gt"))
    {}

    template <immutable_tensor_t<T> Input1>
    auto
    operator()(Input1 input, T value)
    {
        return _M_kernel.template operator()<bool, Input1>(input, value);
    }
};


} // namespace kernel
} // namespace metalchat

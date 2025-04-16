#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/kernel_wrapper.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16> class hadamard {
private:
    inline static const std::string operation_name = "hadamard_" + std::to_string(BlockSize);

    binary_kernel_wrapper<T, BlockSize> _m_kernel;

public:
    hadamard(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return _m_kernel(input1, input2);
    }
};


template <typename T, std::size_t BlockSize = 16> class scalar_mul {
private:
    inline static const std::string operation_name = "scalar_mul_" + std::to_string(BlockSize);

    binary_kernel_wrapper<T, BlockSize> _m_kernel;

public:
    scalar_mul(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input, immutable_scalar_t<T> Multiplier>
    auto
    operator()(Input input, Multiplier multiplier)
    {
        return _m_kernel(input, multiplier);
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, const T multiplier)
    {
        return _m_kernel(input, multiplier);
    }
};


} // namespace kernel
} // namespace metalchat

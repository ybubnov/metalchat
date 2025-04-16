#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/kernel_wrapper.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


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

    kernel_base _m_kernel;

public:
    scalar_mul(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input, immutable_scalar_t<T> Multiplier>
    auto
    operator()(Input input, Multiplier multiplier)
    {
        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.allocator());

        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_back(output_view, input_view, multiplier);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.sizes());
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, const T multiplier)
    {
        return operator()(input, shared_tensor(scalar(multiplier)));
    }
};


} // namespace metalchat

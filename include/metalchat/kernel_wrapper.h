#pragma once

#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize> class binary_kernel_wrapper {
private:
    basic_kernel _m_kernel;

public:
    binary_kernel_wrapper(basic_kernel kernel)
    : _m_kernel(kernel)
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        if (auto size1 = input1.sizes().back(), size2 = input2.sizes().back(); size1 != size2) {
            throw std::invalid_argument(std::format(
                "{}: last dimension should be the same for both tensors {} != {}", _m_kernel.name(),
                size1, size2
            ));
        }

        if (auto numel1 = input1.numel(), numel2 = input2.numel(); numel1 != numel2) {
            throw std::invalid_argument(std::format(
                "{}: data size should be the same for both tensors {} != {}", _m_kernel.name(),
                numel1, numel2
            ));
        }

        auto [grid, thread] = make_kernel_grid_2d(input1, BlockSize);
        auto input1_view = flatten<2>(input1);
        auto input2_view = flatten<2>(input2);
        auto output_view = shared_empty_like<T>(input1_view, _m_kernel.get_allocator());

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input1_view, input2_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input1.shape());
    }

    template <immutable_tensor_t<T> Input1, immutable_scalar_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto input_view = flatten<2>(input1);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.get_allocator());

        auto [grid, thread] = make_kernel_grid_2d(input1, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, input2);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input1.shape());
    }

    template <immutable_tensor_t<T> Input1>
    auto
    operator()(Input1 input1, const T input2)
    {
        return operator()(input1, shared_tensor(scalar(input2)));
    }
};


} // namespace metalchat

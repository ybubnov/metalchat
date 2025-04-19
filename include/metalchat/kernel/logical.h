#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 32> class gt {
private:
    inline static const std::string operation_name = "gt_" + std::to_string(BlockSize);

    basic_kernel _m_kernel;

public:
    gt(hardware_accelerator& gpu)
    : _m_kernel(gpu.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input1>
    auto
    operator()(Input1 input, T value)
    {
        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<bool>(input_view, _m_kernel.get_allocator());

        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, shared_tensor(scalar(value)));

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat

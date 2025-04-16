#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class silu {
private:
    inline static const std::string operation_name = "silu_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    silu(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.allocator());

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.sizes());
    }
};


} // namespace metalchat

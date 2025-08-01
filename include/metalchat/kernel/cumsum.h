#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16> class cumsum {
private:
    basic_kernel _M_kernel;

public:
    cumsum(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("cumsum"))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _M_kernel.get_allocator());

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat

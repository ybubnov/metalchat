#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 32> class roll {
private:
    inline static const std::string operation_name = "roll_" + std::to_string(BlockSize);

    basic_kernel _m_kernel;

public:
    roll(hardware_accelerator& accelerator)
    : _m_kernel(accelerator.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, int32_t shift, int32_t dim)
    {
        auto input_view = flatten<1>(input);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.get_allocator());

        auto input_numel = input_view.numel();
        auto thread_size = std::min(input_numel, _m_kernel.max_threads_per_threadgroup());
        auto thread = dim3(thread_size);
        auto grid = dim3(ceil_div(input_numel, thread_size) * thread_size);

        if (shift < 0) {
            shift = shift + int32_t(input.size(dim));
        }

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(
            output_view, input_view, shared_tensor(scalar<int32_t>(shift)),
            shared_tensor(scalar<int32_t>(input.size(dim))),
            shared_tensor(scalar<int32_t>(input.stride(dim)))
        );

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat

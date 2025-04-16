#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16> class rmsnorm {
private:
    inline static const std::string operation_name = "rmsnorm_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    rmsnorm(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor1_t<T> Weight>
    auto
    operator()(Input input, Weight weight, const float eps = 1e-5)
    {
        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        if (dim_size != weight.size(0)) {
            throw std::invalid_argument(std::format(
                "kernel::rmsnorm: dimension of the input should match weight size {} != {}",
                dim_size, weight.size(0)
            ));
        }

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.allocator());

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future
            = task.bind_front(output_view, input_view, weight, shared_tensor(scalar<float>(eps)));

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.sizes());
    }
};


} // namespace kernel
} // namespace metalchat

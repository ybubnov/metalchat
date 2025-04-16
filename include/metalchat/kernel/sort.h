#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 32> class sort {
private:
    inline static const std::string operation_name = "sort_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    sort(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        using indices_tensor = tensor<int32_t, 2, device_ref<int32_t>>;

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        auto input_view = input.view({-1, int(dim_size)});

        auto values = shared_tensor(empty_like(input_view, _m_kernel.device()));
        auto values_buffer = shared_tensor(empty_like(input_view, _m_kernel.device()));
        auto indices = shared_tensor(indices_tensor(input_view.sizes(), _m_kernel.device()));
        auto indices_buffer = shared_tensor(indices_tensor(input_view.sizes(), _m_kernel.device()));

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);
        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(values, values_buffer, indices, indices_buffer, input_view);

        auto values_future = future_tensor(values, std::move(fn)).view(input.sizes());
        auto indices_future = future_tensor(indices, values_future).view(input.sizes());

        return std::make_tuple(values_future, indices_future);
    }
};


} // namespace metalchat

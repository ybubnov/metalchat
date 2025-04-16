#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class cumsum {
private:
    inline static const std::string operation_name = "cumsum_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    cumsum(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto input_view = input.view({-1, int(dim_size)});

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input_view);

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input.sizes());
    }
};


} // namespace metalchat

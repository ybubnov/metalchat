#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T> class silu {
private:
    inline static const std::string operation_name = "silu";

    kernel_base _m_kernel;

public:
    silu(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(shared_tensor<T, N, InputContainer> input)
    {
        constexpr std::size_t block_size = 32;

        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;
        auto input_view = input.view({-1, int(dim_size)});

        auto [grid, thread] = make_kernel_grid_1d(input, block_size);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input_view);

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input.sizes());
    }
};


} // namespace metalchat

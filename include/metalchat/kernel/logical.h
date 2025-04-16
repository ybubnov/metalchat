#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 32> class gt {
private:
    inline static const std::string operation_name = "gt_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    gt(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto data_size = input1.numel();
        auto dim_size = input1.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = input2.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "gt: last dimension should be the same for both tensors {} != {}", dim_size,
                dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "gt: data size should be the same for both tensors {} != {}", data_size, data_size2
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input1, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1.template flatten<2>(), input2.template flatten<2>());

        auto output = empty_future<bool>({num_rows, dim_size}, std::move(fn));
        return output.view(input1.sizes());
    }
};


} // namespace metalchat

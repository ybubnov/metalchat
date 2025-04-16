#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 32> class add {
private:
    inline static const std::string operation_name = "add_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    add(device& device)
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
                "add: last dimension should be the same for both tensors {} != {}", dim_size,
                dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "add: data size should be the same for both tensors {} != {}", data_size, data_size2
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input1, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1.template flatten<2>(), input2.template flatten<2>());

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input1.sizes());
    }
};


template <typename T> class add2 {
private:
    inline static const std::string operation_name = "add2";

    kernel_base _m_kernel;

public:
    add2(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor2_t<T> Input2>
    requires(Input1::dim() >= 2)
    auto
    operator()(Input1 input1, Input2 input2)
    {
        constexpr std::size_t block_size_x = 4;
        constexpr std::size_t block_size_y = 4;

        constexpr auto M = Input1::dim();

        auto data_size = input1.numel();
        auto dim0_size = input2.size(0);
        auto dim1_size = input2.size(1);
        auto num_rows = data_size / (dim0_size * dim1_size);

        if (dim0_size != input1.size(M - 2) || dim1_size != input1.size(M - 1)) {
            throw std::invalid_argument(std::format(
                "add2: last dimensions should be the same for both tensors {}x{} != {}x{}",
                input1.size(M - 2), input1.size(M - 1), dim0_size, dim1_size
            ));
        }

        auto input1_view = input1.view({-1, int(dim0_size), int(dim1_size)});

        auto thread_size_x = ceil_div(dim0_size, block_size_x);
        auto thread_size_y = ceil_div(dim1_size, block_size_y);
        auto thread = dim3(thread_size_x, thread_size_y);
        auto grid = dim3(thread_size_x * num_rows, thread_size_y);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1_view, input2);

        auto output = empty_future<T>({num_rows, dim0_size, dim1_size}, std::move(fn));
        return output.view(input1.sizes());
    }
};


template <typename T, std::size_t BlockSize = 32> class sub {
private:
    inline static const std::string operation_name = "sub_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    sub(device& device)
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
                "sub: last dimension should be the same for both tensors {} != {}", dim_size,
                dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "sub: data size should be the same for both tensors {} != {}", data_size, data_size2
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input1, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1.template flatten<2>(), input2.template flatten<2>());

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input1.sizes());
    }
};

} // namespace metalchat

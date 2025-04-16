#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class hadamard {
private:
    inline static const std::string operation_name = "hadamard_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    hadamard(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor Input1Tensor, immutable_tensor Input2Tensor>
    auto
    operator()(Input1Tensor input1, Input2Tensor input2)
    {
        auto data_size = input1.numel();
        auto dim_size = input1.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = input2.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "kernel::hadamard: last dimension size should be the same for both tensors {} != "
                "{}",
                dim_size, dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "kernel::hadamard: data size should be the same for both tensors {} != {}",
                data_size, data_size2
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input1, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1.template flatten<2>(), input2.template flatten<2>());

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input1.sizes());
    }
};


template <typename T, std::size_t BlockSize = 16> class scalar_mul {
private:
    inline static const std::string operation_name = "scalar_mul_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    scalar_mul(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor InputTensor, immutable_scalar Multiplier>
    auto
    operator()(InputTensor input, Multiplier multiplier)
    {
        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;
        auto input_view = input.view({-1, int(dim_size)});

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input_view, multiplier);

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input.sizes());
    }

    template <immutable_tensor InputTensor>
    auto
    operator()(InputTensor input, const T multiplier)
    {
        return operator()(input, shared_tensor(scalar(multiplier)));
    }
};


} // namespace metalchat

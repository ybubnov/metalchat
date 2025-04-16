#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class cpy {
private:
    inline static const std::string operation_name = "copy_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

    template <immutable_tensor2_t<T> Input, immutable_tensor2_t<T> Output>
    auto
    copy(Input input, Output output)
    {
        if (auto dim_size = output.sizes().back(); dim_size != input.sizes().back()) {
            throw std::invalid_argument(std::format(
                "kernel::copy: last dimension should be the same for both tensors {} != {}",
                input.sizes().back(), dim_size
            ));
        }

        if (auto data_size = output.numel(); data_size != input.numel()) {
            throw std::invalid_argument(std::format(
                "kernel::copy: data size should be the same for both tensors {} != {}",
                input.sizes().back(), data_size
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(output, input);

        return future_tensor(output, std::move(fn));
    }

public:
    cpy(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    /// Copy values from input to the output.
    ///
    /// The metal kernel implementation supports only copying of 2-dimensional tensors,
    /// considering that all dimensions that are larger than 1 (a vector) are simply batch
    /// dimensions, we could simply collapse all of them into a single batch dimension.
    ///
    /// The resulting tensor from the future operation is also 2-dimensional, therefore
    /// if caller wants to retain original dimensionality, she must keep the original
    /// output tensor.
    ///
    /// The operation is executed asynchronously on GPU, therefore output tensor should be
    /// allocated on GPU memory.
    template <immutable_tensor_t<T> Input, immutable_device_tensor_t<T> Output>
    auto
    operator()(Input input, Output output)
    {
        return copy(input.template flatten<2>(), output.template flatten<2>());
    }
};


template <typename T, std::size_t BlockSize = 16> class inplace_index_set {
private:
    inline static const std::string operation_name
        = "inplace_index_set_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    inplace_index_set(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor_t<bool> Mask>
    auto
    operator()(Input input, Mask mask, T value)
    {
        // TODO: ensure that input is the same shape as mask.
        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto input_view = input.template flatten<2>();
        auto mask_view = mask.template flatten<2>();

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(input_view, mask_view, shared_tensor(scalar(value)));

        auto output = future_tensor(input, std::move(fn));
        return output.view(input.sizes());
    }
};


} // namespace metalchat

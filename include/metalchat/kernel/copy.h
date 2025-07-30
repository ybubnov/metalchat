#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16> class cpy {
private:
    basic_kernel _m_kernel;

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

        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output, input);

        return future_tensor(output, std::move(task_future));
    }

public:
    cpy(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("copy"))
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
    template <immutable_tensor_t<T> Input, immutable_hardware_tensor_t<T> Output>
    auto
    operator()(Input input, Output output)
    {
        return copy(flatten<2>(input), flatten<2>(output));
    }
};


template <typename T, std::size_t BlockSize = 16> class scatter {
private:
    basic_kernel _m_kernel;

public:
    scatter(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("scatter"))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor_t<bool> Mask>
    auto
    operator()(Input input, Mask mask, T value)
    {
        // TODO: ensure that input is the same shape as mask.
        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto input_view = flatten<2>(input);
        auto mask_view = flatten<2>(mask);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(input_view, mask_view, scalar(value));

        auto output = future_tensor(input, std::move(task_future));
        return output.view(input.shape());
    }
};


template <typename T, std::size_t BlockSize = 16> class gather {
private:
    basic_kernel _m_kernel;

public:
    gather(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("gather"))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor_t<int32_t> Index>
    auto
    operator()(Input input, Index index)
    {
        // TODO:: ensure that input has the same dimensions as index.
        auto [grid, thread] = make_kernel_grid_2d(index, BlockSize);

        auto input_view = flatten<2>(input);
        auto index_view = flatten<2>(index);
        auto output_view = shared_empty_like<T>(index_view, _m_kernel.get_allocator());

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, index_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(index.shape());
    }
};


} // namespace kernel
} // namespace metalchat

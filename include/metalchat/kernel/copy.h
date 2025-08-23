#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


/// Create a copy of a tensor.
///
/// The metal kernel implementation supports only copying of 2-dimensional tensors,
/// considering that all dimensions that are larger than 1 (a vector) are simply batch
/// dimensions, we could simply collapse all of them into a single batch dimension.
///
/// The resulting tensor from the future operation is also 2-dimensional, therefore
/// if caller wants to retain original dimensionality, she must keep the original
/// output tensor or adjust the resulting tensor shape as needed.
///
/// \note The operation is executed asynchronously on GPU, therefore output tensor should
/// be allocated on GPU memory beforehand.
template <typename T, std::size_t BlockSize = 16> class clone {
private:
    basic_kernel _M_kernel;

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

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, input);

        return future_tensor(output, std::move(task_future));
    }

public:
    /// The kernel constructor.
    clone(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T, BlockSize>("copy"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input a tensor to clone data from.
    /// \param output a tensor to clone data to.
    template <immutable_tensor_t<T> Input, immutable_hardware_tensor_t<T> Output>
    auto
    operator()(Input input, Output output)
    {
        return copy(flatten<2>(input), flatten<2>(output));
    }
};


/// Writes values into the tensor at the specified indices.
///
/// \warning When indices are not unique, the behaviour is non-deterministic.
template <typename T, std::size_t BlockSize = 16> class scatter {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    scatter(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("scatter"))
    {}

    /// Invokes the kernel, and writes a single value to the output tensor according to
    /// the specified boolean mask.
    ///
    /// \param output a tensor to write data to.
    /// \param mask a boolean mask tensor (should be the same size as an output tensor).
    /// \param value a value to write.
    ///
    /// ```c++
    /// auto T = tensor<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    /// auto M = tensor<bool>({{true, false, false}, {false, true, true}});
    ///
    /// auto accelerator = hardware_accelerator();
    /// auto scatter = kernel::scatter(accelerator);
    ///
    /// auto output = scatter(T, M, 9.0);
    /// std::cout << output.get() << std::endl;
    /// // out:
    /// // [[9.0, 2.0, 3.0],
    /// //  [4.0, 9.0, 9.0]], sizes=(2, 3)
    /// ```
    template <immutable_tensor_t<T> Output, immutable_tensor_t<bool> Mask>
    auto
    operator()(Output output, Mask mask, T value)
    {
        // TODO: ensure that input is the same shape as mask.
        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto output_view = flatten<2>(output);
        auto mask_view = flatten<2>(mask);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, mask_view, scalar(value));

        auto result = future_tensor(output, std::move(task_future));
        return result.view(input.shape());
    }
};


/// Gathers values given the index tensor.
///
/// ```c++
/// auto T = tensor<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
/// auto index = tensor<int32_t>({{0, 0}, {1, 0}});
///
/// auto accelerator = hardware_accelerator();
/// auto gather = kernel::gather(accelerator);
///
/// auto output = gather(T, index);
/// std::cout << output.get() << std::endl;
/// // out:
/// // [[1.0, 1.0],
/// //  [5.0, 4.0]], sizes=(2, 2)
/// ```
///
/// \note Current implementation treats all tensors as 2-dimensional with dimension 0 as a batch
/// dimension, and gather elements only along 0 dimension.
template <typename T, std::size_t BlockSize = 16> class gather {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    gather(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("gather"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input an input tensor to gather values from.
    /// \param index an index tensor that specifies locations of elements within input tensor.
    ///
    /// \return a \ref future_tensor with the elements gathered from an input tensor.
    template <immutable_tensor_t<T> Input, immutable_tensor_t<int32_t> Index>
    auto
    operator()(Input input, Index index)
    {
        // TODO:: ensure that input has the same dimensions as index.
        auto [grid, thread] = make_kernel_grid_2d(index, BlockSize);

        auto input_view = flatten<2>(input);
        auto index_view = flatten<2>(index);
        auto output_view = shared_empty_like<T>(index_view, _M_kernel.get_allocator());

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, index_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(index.shape());
    }
};


} // namespace kernel
} // namespace metalchat

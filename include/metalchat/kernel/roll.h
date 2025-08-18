#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


/// Roll the tensor along the given dimension. Elements that are shifted beyond the last
/// position are re-introduced at the first position. The tensor is always flattened before
/// rolling and then restored to the original shape.
template <typename T, std::size_t BlockSize = 32> class roll {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    roll(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T, BlockSize>("roll"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input an input tensor.
    /// \param shift the number of places by which the elements of the tensor are shifted.
    /// \param dim an axis along which to roll.
    ///
    /// \return a tensor with elements rolled along the specified dimension.
    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, int32_t shift, std::size_t dim)
    {
        auto output = shared_empty_like<T>(input, _M_kernel.get_allocator());
        return operator()(input, output, shift, dim);
    }

    /// Invokes the kernel.
    ///
    /// \param input an input tensor.
    /// \param output an output tensor.
    /// \param shift the number of places by which the elements of the tensor are shifted.
    /// \param dim an axis along which to roll.
    ///
    /// \return a tensor with elements rolled along the specified dimension.
    template <immutable_tensor_t<T> Input, immutable_tensor_t<T> Output>
    auto
    operator()(Input input, Output output, int32_t shift, std::size_t dim)
    {
        auto input_view = flatten<1>(input);
        auto output_view = flatten<1>(output);

        // The roll kernel does not assume any concrete shape of the input tensor so that
        // implementation could roll any dimensions (including bath dimension).
        //
        // This implies that regular methods for kernel grid creation (maker_kernel_grid_1d,
        // and make_kernel_grid_2d) won't fit our needs. So schedule a grid that allocates
        // as large threads as possible (or as needed, depending on the tensor sizes).
        auto input_numel = input_view.numel();
        auto thread_size = std::min(input_numel, _M_kernel.max_threads_per_threadgroup());
        auto thread = dim3(thread_size);
        auto grid = dim3(ceil_div(input_numel, thread_size) * thread_size);

        shift = shift % input.size(dim);
        if (shift < 0) {
            shift = int32_t(input.size(dim)) + shift;
        }

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(
            output_view, input_view, scalar<int32_t>(shift), scalar<int32_t>(input.size(dim)),
            scalar<int32_t>(input.stride(dim))
        );

        return future_tensor(output, std::move(task_future));
    }
};


} // namespace kernel
} // namespace metalchat

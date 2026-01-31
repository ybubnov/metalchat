// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <format>
#include <future>
#include <tuple>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/future.h>


namespace metalchat {


inline std::size_t
ceil_div(std::size_t a, std::size_t b)
{
    return (a + b - 1) / b;
}


inline std::size_t
ceil_pow2(std::size_t value)
{
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}


std::tuple<dim3, dim3>
make_kernel_grid_2d(std::size_t num_rows, std::size_t dim_size, std::size_t max_threads);


template <immutable_tensor Tensor>
std::tuple<dim3, dim3>
make_kernel_grid_2d(const Tensor& t, std::size_t max_threads)
{
    auto data_size = t.numel();
    auto dim_size = t.sizes().back();
    auto num_rows = data_size / dim_size;

    return make_kernel_grid_2d(num_rows, dim_size, max_threads);
}


class basic_kernel {
private:
    std::string _M_name;
    metal::shared_kernel _M_kernel;
    hardware_accelerator _M_accelerator;

public:
    basic_kernel(metal::shared_kernel kernel, const hardware_accelerator& accelerator);

    std::string
    name() const;

    const metal::shared_kernel
    get_metal_kernel() const;

    hardware_accelerator::allocator_type
    get_allocator() const;

    hardware_accelerator&
    get_accelerator();

    std::size_t
    max_threads_per_threadgroup();
};


/// A class that wraps a Metal kernel task, arguments for the task, and scheduler parameters.
///
/// Tasks are executed asynchronously on a \ref hardware_accelerator. This implies that before
/// scheduling a task execution all specified arguments must be bound to the task using either
/// \ref kernel_task::bind_front or \ref bind_back methods.
///
/// \warning Usually there is no need to create a kernel task manually, as a kernel usually
/// creates one for you and passes required arguments correctly. You could explore a
/// collection of available kernels in \verbatim embed:rst:inline :doc:`metal` \endverbatim.
///
/// Most commonly, task are used as asynchronously invocable instances for \ref future_tensor,
/// so that operation that produces result for a tensor could be asynchronously awaited.
///
/// \warning When kernel task is used with \ref future_tensor, consider moving the ownership of
/// the task to the future tensor with the respective constructor and `std::move` to release
/// memory from the dependent tensors (kernel task arguments) on a kernel completion.
template <immutable_tensor... Args> class kernel_task {
private:
    using arguments_type = std::tuple<Args...>;

    basic_kernel _M_kernel;
    std::shared_ptr<kernel_thread> _M_this_thread_ptr;
    std::shared_ptr<arguments_type> _M_args;

    /// Configuration of the Metal grid to invoke this particular kernel. Grid specifies
    /// the total number of threads in a grid, while thread defines a number of threads
    /// in a threadgroup.
    dim3 _M_grid;
    dim3 _M_thread;

public:
    /// The copy constructor of the \ref kernel_task.
    kernel_task(const kernel_task& task) noexcept = default;

    kernel_task(basic_kernel kernel, dim3 grid, dim3 thread, std::tuple<Args...>&& args)
    : _M_kernel(kernel),
      _M_this_thread_ptr(nullptr),
      _M_args(std::make_shared<arguments_type>(args)),
      _M_grid(grid),
      _M_thread(thread)
    {
        auto max_threads = kernel.max_threads_per_threadgroup();
        if (thread.numel() > max_threads) {
            throw std::invalid_argument(std::format(
                "kernel: `{}` <{}, {}, {}> configuration exceeds maximum number of threads per "
                "group {}",
                kernel.name(), thread.x, thread.y, thread.z, max_threads
            ));
        }

        if (grid.numel() < thread.numel()) {
            throw std::invalid_argument(std::format(
                "kernel: there are less threads in grid <{}, {}, {}> than in group <{}, {}, {}>",
                grid.x, grid.y, grid.z, thread.x, thread.y, thread.z
            ));
        }
    }

    /// Creates a new kernel task with the specified kernel function and hardware grid
    /// configuration.
    ///
    /// \param kernel a kernel function instance.
    /// \param grid total size of 3-dimensional GPU compute grid.
    /// \param thread a size of 3-dimensional GPU compute thread group.
    /// \param args optional kernel arguments.
    ///
    /// ```c++
    /// auto accelerator = hardware_accelerator();
    /// auto kernel = accelerator.load<float, 16>("hadamard");
    ///
    /// // Create a kernel with 4 thread groups of size 16x16x1 each.
    /// auto task = kernel_task(kernel, dim3(64, 64), dim3(16, 16));
    /// ```
    kernel_task(basic_kernel kernel, dim3 grid, dim3 thread, Args... args)
    : kernel_task(kernel, grid, thread, std::make_tuple(args...))
    {}

    /// Schedules execution of the stored kernel. The method returns a shared future that could
    /// be used to await for the task completion.
    ///
    /// \warning The function call operator can be called only once for each `kernel_task`.
    std::shared_future<void>
    operator()()
    {
        if (_M_this_thread_ptr != nullptr) {
            throw std::runtime_error(std::format(
                "kernel_task: the kernel '{}' has already been invoked", _M_kernel.name()
            ));
        }

        _M_this_thread_ptr = _M_kernel.get_accelerator().get_this_thread();
        return _M_this_thread_ptr->push(*this);
    }

    /// Schedules execution of the stored kernel. The method returns a shared future that could
    /// be used to await for the task completion.
    ///
    /// This method allows to specify an arbitrary callback function that will be executed before
    /// releasing the returned future object.
    ///
    /// \warning The function call operator can be called only once for each `kernel_task`.
    std::shared_future<void>
    operator()(std::function<void()> callback)
    {
        if (_M_this_thread_ptr != nullptr) {
            throw std::runtime_error(std::format(
                "kernel_task: the kernel '{}' has already been invoked", _M_kernel.name()
            ));
        }

        _M_this_thread_ptr = _M_kernel.get_accelerator().get_this_thread();
        return _M_this_thread_ptr->push(*this, callback);
    }

    /// Encode the kernel and all it's arguments with the specified encoder.
    ///
    /// The encoding process implies setup of kernel arguments (tensors), data offsets, and
    /// kernel dependencies (outputs from other kernels).
    ///
    /// \note This method is called by a `kernel_thread`, when the kernel is scheduled for
    /// executions by calling one of overloaded function call operators
    /// \ref kernel_task::operator()(std::function<void()>), or \ref kernel_task::operator()()
    /// therefore there is no need to call this method manually.
    void
    encode(hardware_function_encoder encoder)
    {
        return encode(encoder, std::index_sequence_for<Args...>{});
    }

    /// Immideately schedules execution of the kernel task by a hardware accelerator.
    ///
    /// The accelerator keeps a queue of tasks and executes them in batches, so once a batch is
    /// assembled, accelerator starts processing it. This behaviour could be changed by calling
    /// this method, so that processing starts for all tasks in the buffer.
    ///
    /// Method raises `std::runtime_error`, when this method is executed for a task that is
    /// not invoked (pushed to a command buffer) with one of \ref kernel_task::operator()(), or
    /// \ref kernel_task::operator()(std::function<void()>) methods.
    void
    make_ready_at_thread_exit()
    {
        if (_M_this_thread_ptr == nullptr) {
            throw std::runtime_error(
                std::format("kernel_task: kernel '{}' was not invoked", _M_kernel.name())
            );
        }

        _M_this_thread_ptr->make_ready_at_thread_exit();
        _M_args.reset();
    }

    /// Returns a new kernel task with bound arguments at positions starting from the beginning
    /// of the task arguments sequence.
    ///
    /// The kernel task expects all arguments to be tensors, since the kernel should be encodable
    /// to the hardware kernel queue.
    ///
    /// \tparam FrontArgs a sequence of argument types to bind.
    /// \param front_args a sequence of arguments to bind.
    ///
    /// \note The bound arguments are shallow copies of the tensor, meaning that tensor layout
    /// (sizes, strides, offsets) are preserved, but data might be modified through the tensor
    /// that shares the same underlying contiguous container.
    template <immutable_tensor... FrontArgs>
    kernel_task<FrontArgs..., Args...>
    bind_front(FrontArgs... front_args)
    {
        return kernel_task<FrontArgs..., Args...>(
            _M_kernel, _M_grid, _M_thread, std::tuple_cat(std::make_tuple(front_args...), *_M_args)
        );
    }

    /// Returns a new kernel task with bound arguments appended to the end of the task arguments
    /// sequence.
    ///
    /// The kernel task expects all arguments to be tensors, since the kernel should be encodable
    /// to the hardware kernel queue.
    ///
    /// \tparam BackArgs a sequence of argument types to bind.
    /// \param back_args a sequence of arguments to bind.
    template <immutable_tensor... BackArgs>
    kernel_task<Args..., BackArgs...>
    bind_back(BackArgs... back_args)
    {
        return kernel_task<Args..., BackArgs...>(
            _M_kernel, _M_grid, _M_thread, std::tuple_cat(*_M_args, std::make_tuple(back_args...))
        );
    }

    /// Returns a name of the kernel.
    std::string
    name() const
    {
        return _M_kernel.name();
    }

protected:
    template <std::size_t... Indices>
    void
    encode(hardware_function_encoder encoder, std::index_sequence<Indices...>)
    {
        encoder.initialize(_M_kernel.name(), _M_kernel.get_metal_kernel());

        ([&] {
            using tensor_type = std::tuple_element<Indices, arguments_type>::type;
            using value_type = tensor_type::value_type;

            const auto& arg = std::get<Indices>(*_M_args);
            encoder.encode<value_type>(arg);
        }(), ...);

        encoder.dispatch(_M_grid, _M_thread);
    }
};


template <typename T> class binary_kernel_wrapper {
private:
    basic_kernel _M_kernel;

public:
    binary_kernel_wrapper(basic_kernel kernel)
    : _M_kernel(kernel)
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return operator()<T, Input1, Input2>(input1, input2);
    }

    std::string
    name() const
    {
        return _M_kernel.name();
    }

    template <typename R, immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        if (auto size1 = input1.sizes().back(), size2 = input2.sizes().back(); size1 != size2) {
            throw std::invalid_argument(std::format(
                "{}: last dimension should be the same for both tensors {} != {}", _M_kernel.name(),
                size1, size2
            ));
        }

        if (auto numel1 = input1.numel(), numel2 = input2.numel(); numel1 != numel2) {
            throw std::invalid_argument(std::format(
                "{}: data size should be the same for both tensors {} != {}", _M_kernel.name(),
                numel1, numel2
            ));
        }

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_kernel_grid_2d(input1, max_threads);
        auto input1_view = flatten<2>(input1);
        auto input2_view = flatten<2>(input2);
        auto output_view = shared_empty_like<R>(input1_view, _M_kernel.get_allocator());

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input1_view, input2_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input1.shape());
    }

    template <immutable_tensor_t<T> Input1, immutable_scalar_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return operator()<T, Input1, Input2>(input1, input2);
    }

    template <immutable_tensor_t<T> Input1>
    auto
    operator()(Input1 input1, T input2)
    {
        return operator()(input1, scalar(input2));
    }

    template <typename R, immutable_tensor_t<T> Input1, immutable_scalar_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto input_view = flatten<2>(input1);
        auto output_view = shared_empty_like<R>(input_view, _M_kernel.get_allocator());

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_kernel_grid_2d(input1, max_threads);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, input2);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input1.shape());
    }

    template <typename R, immutable_tensor_t<T> Input1>
    auto
    operator()(Input1 input1, T input2)
    {
        auto input2_ = scalar(input2);
        using Input2 = decltype(input2_);

        return operator()<R, Input1, Input2>(input1, input2_);
    }
};


} // namespace metalchat

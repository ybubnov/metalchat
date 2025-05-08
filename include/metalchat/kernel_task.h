#pragma once

#include <future>

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/future.h>


namespace metalchat {


inline std::size_t
ceil_div(std::size_t a, std::size_t b)
{
    return (a + b - 1) / b;
}


template <immutable_tensor Tensor>
std::tuple<dim3, dim3>
make_kernel_grid_1d(const Tensor& t, std::size_t block_size)
{
    auto data_size = t.numel();
    auto dim_size = t.sizes().back();
    auto num_rows = data_size / dim_size;

    auto thread_size = ceil_div(dim_size, block_size);
    auto thread = dim3(thread_size);
    auto grid = dim3(thread_size * num_rows);

    return std::forward_as_tuple(grid, thread);
}


template <immutable_tensor Tensor>
std::tuple<dim3, dim3>
make_kernel_grid_2d(const Tensor& t, std::size_t block_size)
{
    auto data_size = t.numel();
    auto dim_size = t.sizes().back();
    auto num_rows = data_size / dim_size;

    auto thread_size = ceil_div(dim_size, block_size);
    auto thread = dim3(thread_size);
    auto grid = dim3(thread_size * num_rows, block_size);

    return std::forward_as_tuple(grid, thread);
}


template <immutable_tensor... Args> class kernel_task {
private:
    using arguments_type = std::tuple<Args...>;

    basic_kernel _m_kernel;
    std::shared_ptr<kernel_thread> _m_this_thread_ptr;

    arguments_type _m_args;

    /// Configuration of the Metal grid to invoke this particular kernel. Grid specifies
    /// the total number of threads in a grid, while thread defines a number of threads
    /// in a threadgroup.
    dim3 _m_grid;
    dim3 _m_thread;

public:
    kernel_task(const kernel_task& task) noexcept = default;

    kernel_task(basic_kernel kernel, dim3 grid, dim3 thread, std::tuple<Args...>&& args)
    : _m_kernel(kernel),
      _m_this_thread_ptr(nullptr),
      _m_args(args),
      _m_grid(grid),
      _m_thread(thread)
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

    kernel_task(basic_kernel kernel, dim3 grid, dim3 thread, Args... args)
    : kernel_task(kernel, grid, thread, std::make_tuple(args...))
    {}

    std::shared_future<void>
    operator()()
    {
        if (_m_this_thread_ptr != nullptr) {
            throw std::runtime_error(std::format("kernel_task: the kernel has already been invoked")
            );
        }

        _m_this_thread_ptr = _m_kernel.get_this_thread();
        return _m_this_thread_ptr->push(*this);
    }

    std::shared_future<void>
    operator()(std::function<void()> callback)
    {
        if (_m_this_thread_ptr != nullptr) {
            throw std::runtime_error(std::format("kernel_task: the kernel has already been invoked")
            );
        }

        _m_this_thread_ptr = _m_kernel.get_this_thread();
        return _m_this_thread_ptr->push(*this, callback);
    }

    void
    encode(hardware_function_encoder encoder)
    {
        return encode(encoder, std::index_sequence_for<Args...>{});
    }

    template <std::size_t... Indices>
    void
    encode(hardware_function_encoder encoder, std::index_sequence<Indices...>)
    {
        encoder.initialize(_m_kernel.name(), _m_kernel.pipeline());

        ([&] {
            using tensor_type = std::tuple_element<Indices, arguments_type>::type;
            using value_type = tensor_type::value_type;

            const auto& arg = std::get<Indices>(_m_args);
            encoder.encode<value_type>(arg);
        }(), ...);

        encoder.dispatch(_m_grid, _m_thread);
    }

    void
    make_ready_at_thread_exit()
    {
        if (_m_this_thread_ptr != nullptr) {
            _m_this_thread_ptr->make_ready_at_thread_exit();
        } else {
            throw std::runtime_error(std::format("kernel_task: task was not invoked"));
        }
    }

    template <immutable_tensor... FrontArgs>
    kernel_task<FrontArgs..., Args...>
    bind_front(FrontArgs... front_args)
    {
        return kernel_task<FrontArgs..., Args...>(
            _m_kernel, _m_grid, _m_thread, std::tuple_cat(std::make_tuple(front_args...), _m_args)
        );
    }

    template <immutable_tensor... BackArgs>
    kernel_task<Args..., BackArgs...>
    bind_back(BackArgs... back_args)
    {
        return kernel_task<Args..., BackArgs...>(
            _m_kernel, _m_grid, _m_thread, std::tuple_cat(_m_args, std::make_tuple(back_args...))
        );
    }

    std::string
    name()
    {
        return _m_kernel.name();
    }
};


} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <future>
#include <mutex>

#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/shared.h>


namespace metalchat {


/// The `asynchronously_invocable` concept specifies that a type T can be called asynchronously,
/// and result could be either awaited through future or through callback (or both).
///
/// Effectively, this type is used as a opaque wrapper around a task that computes result of a
/// future tensor. The implementation of `operator()` should not block the thread and exit fast.
/// The implementation of `make_ready_at_thread_exit` should block the thread, until the result
/// is not computed.
template <typename T>
concept asynchronously_invocable = requires(std::remove_reference_t<T> t) {
    { t() } -> std::same_as<std::shared_future<void>>;
    { t(std::function<void()>()) } -> std::same_as<std::shared_future<void>>;
    { t.make_ready_at_thread_exit() } -> std::same_as<void>;
};


/// A tensor associated with a computation task, which result is not ready yet.
///
/// A future tensor holds a pointer to the pre-allocated on-device memory, and a task that
/// is responsible for filling that memory. Future tensor is immutable, meaning that it's
/// content cannot be modified before the completion of an associated task.
///
/// Since the tensor is immutable, any immutable operation (which does not modify the
/// underlying data) could be executed. Such operations include: transposition, slicing,
/// narrowing, dimensionality expansion, etc.
template <typename T, std::size_t N> class future_tensor {
public:
    using result_type = shared_tensor<T, N, hardware_memory_container<T>>;

    using future_type = std::shared_ptr<std::function<void()>>;

    using future_mutex_type = std::shared_ptr<std::mutex>;

    using future_wait_type = std::shared_ptr<std::function<void()>>;

    /// Alias of the tensor type.
    using value_type = result_type::value_type;

    /// Pointer to the tensor type.
    using pointer_type = result_type::pointer_type;

    using accessor_type = tensor_accessor;

    /// Container type storing the data of the tensor.
    using container_type = result_type::container_type;

    /// Pointer to the container type storing the data of the tensor.
    using container_pointer = result_type::container_pointer;

    /// Contiguous iterator of the tensor data.
    using iterator = result_type::iterator;

    /// Contiguous constant iterator of the tensor data.
    using const_iterator = result_type::const_iterator;

    template <asynchronously_invocable Task>
    future_tensor(result_type result, Task&& task)
    : _M_result(result),
      _M_future_mutex(std::make_shared<std::mutex>())
    {
        const std::scoped_lock __lock(*_M_future_mutex);

        // Enqueue the calculations to compute the tensor, upon the completion,
        // command buffers calls a callback that sets a value to the promise, so
        // all waiting routines will be unblocked.
        //
        // The main advantage of this approach is that the task and all its associated
        // memory will be released as a result of calling this callback.
        auto future = task([ft = std::make_shared<future_tensor<T, N>>(*this)] {
            const std::scoped_lock __lock(*(ft->_M_future_mutex));

            ft->_M_future_wait = nullptr;
            ft->_M_future = nullptr;
        });

        _M_future = std::make_shared<future_type::element_type>(
            std::bind(&std::shared_future<void>::wait, std::move(future))
        );

        // Erase type of the task, and simply ensure that task is ready, when a user
        // calls either `wait` or `get` method of the future tensor.
        _M_future_wait = std::make_shared<future_wait_type::element_type>(
            std::bind(&Task::make_ready_at_thread_exit, std::move(task))
        );
    }

    template <typename U, std::size_t M>
    future_tensor(result_type result, future_tensor<U, M> future)
    : _M_result(result),
      _M_future_mutex(std::make_shared<std::mutex>())
    {
        const std::scoped_lock __lock(*future._M_future_mutex);

        // Wait on the same future.
        _M_future = future._M_future;
        _M_future_wait = std::make_shared<future_wait_type::element_type>(
            [future = std::make_shared<future_tensor<U, M>>(future)] { future->wait(); }
        );
    }

    /// Create a future tensor that expects completion of two other future tensors
    ///
    /// A future tensor `result` becomes the result of the new tensor completion. This
    /// operation in non-destructible, so both `result` and `future` tensors could be
    /// awaited separately from this tensor.
    template <typename U, std::size_t M>
    future_tensor(future_tensor result, future_tensor<U, M> future)
    : _M_result(result._M_result),
      _M_future_mutex(std::make_shared<std::mutex>())
    {
        _M_future = nullptr;
        _M_future_wait = std::make_shared<future_wait_type::element_type>(
            [result = std::make_shared<future_tensor<T, N>>(result),
             future = std::make_shared<future_tensor<U, M>>(future)] {
            result->wait();
            future->wait();
        }
        );
    }

    /// Create a future tensor that expects completion of the specified task.
    ///
    /// A new tensor will wait for completion of both self task and a new asynchronously
    /// invocable task, and only then makes result accessible.
    ///
    /// ```c++
    /// #include <future>
    ///
    /// struct noop_async_func {
    ///     std::promise<void> promise;
    ///
    ///     noop_async_func()
    ///     : promise()
    ///     {
    ///         promise.set_value();
    ///     }
    ///
    ///     std::shared_future<void>
    ///     operator()()
    ///     {
    ///         return std::shared_future(promise.get_future());
    ///     }
    ///
    ///     // In this example, callback is executed immediately since the promise
    ///     // is ready on the class instantiation.
    ///     std::shared_future<void>
    ///     operator()(std::function<void()> callback)
    ///     {
    ///         callback();
    ///         return std::shared_future(promise.get_future());
    ///     }
    ///
    ///     void
    ///     make_ready_at_thread_exit()
    ///     { }
    /// };
    ///
    ///
    /// auto accelerator = hardware_accelerator(16);
    /// auto T = future_tensor(empty<float>({3, 2}, accelerator));
    ///
    /// // Tensor `F` will be waiting for the completion of the specified
    /// // asynchronously_invocable function.
    /// auto F = future_tensor(T, noop_async_func());
    /// ```
    template <asynchronously_invocable Task>
    future_tensor(future_tensor result, Task&& task)
    : future_tensor(result, future_tensor(result._M_result, std::move(task)))
    {}

    /// A naive future tensor that does not wait for any task and returns result immediately.
    future_tensor(result_type result)
    : _M_result(result),
      _M_future_mutex(std::make_shared<std::mutex>()),
      _M_future(nullptr),
      _M_future_wait(nullptr)
    {}

    future_tensor()
    : _M_result(),
      _M_future_mutex(std::make_shared<std::mutex>()),
      _M_future(nullptr),
      _M_future_wait(nullptr)
    {}

    future_tensor(result_type::tensor_type&& result)
    : future_tensor(shared_tensor(std::move(result)))
    {}

    template <immutable_tensor Tensor, hardware_allocator_t<void> Allocator>
    future_tensor(Tensor&& t, Allocator alloc)
    : future_tensor(move(t, alloc))
    {}

    /// Waits for (by calling \ref future_tensor::wait) until the shared tensor is ready, then
    /// retrieves the value stored in the shared state.
    result_type
    get()
    {
        wait();
        return _M_result;
    }

    /// Return the result tensor without waiting of the associated operation completion.
    ///
    /// \warning Since the operation is not awaited, the data container of the returned tensor
    /// could (and will) be populated asynchronously to the main application thread.
    result_type
    get_nowait() const
    {
        return _M_result;
    }

    /// Blocks until the result becomes available.
    void
    wait()
    {
        const std::scoped_lock __lock(*_M_future_mutex);

        if (_M_future_wait) {
            (*_M_future_wait)();
        }

        if (_M_future) {
            (*_M_future)();
        }

        // Once the waiting of the future completion is done, erase associated
        // with this task, but setting an empty function (lambda), so that the
        // task could be destroyed.
        _M_future = nullptr;
        _M_future_wait = nullptr;
    }

    /// See \ref tensor::dim.
    static constexpr std::size_t
    dim()
    {
        return N;
    }

    /// See \ref tensor::numel.
    std::size_t
    numel() const
    {
        return _M_result.numel();
    }

    /// See \ref tensor::accessor.
    const tensor_accessor&
    accessor() const
    {
        return _M_result.accessor();
    }

    /// See \ref tensor::container.
    container_type&
    container() const
    {
        return _M_result.container();
    }

    /// See \ref tensor::container_ptr.
    std::shared_ptr<basic_container>
    container_ptr() const
    {
        return _M_result.container_ptr();
    }

    /// See \ref tensor::data_ptr.
    ///
    /// \warning Tensor must be awaited before accessing data with an iterator.
    pointer_type
    data_ptr()
    {
        return _M_result.data_ptr();
    }

    /// See \ref tensor::data_ptr.
    ///
    /// \warning Tensor must be awaited before accessing data with an iterator.
    const pointer_type
    data_ptr() const
    {
        return _M_result.data_ptr();
    }

    /// See \ref tensor::size.
    std::size_t
    size(std::size_t dim) const
    {
        return _M_result.size(dim);
    }

    /// See \ref tensor::sizes.
    const std::span<std::size_t>
    sizes() const
    {
        return _M_result.sizes();
    }

    /// See \ref tensor::shape.
    const std::span<std::size_t, N>
    shape() const
    {
        return _M_result.shape();
    }

    /// See \ref tensor::stride.
    std::size_t
    stride(std::size_t dim) const
    {
        return _M_result.stride(dim);
    }

    /// See \ref tensor::strides.
    const std::span<std::size_t>
    strides() const
    {
        return _M_result.strides();
    }

    /// See \ref tensor::offset.
    std::size_t
    offset(std::size_t dim) const
    {
        return _M_result.offset(dim);
    }

    /// See \ref tensor::offsets.
    const std::span<std::size_t>
    offsets() const
    {
        return _M_result.offsets();
    }

    /// See \ref tensor::begin.
    ///
    /// \warning Tensor must be awaited before accessing data with an iterator.
    iterator
    begin()
    {
        return _M_result.begin();
    }

    /// See \ref tensor::end.
    iterator
    end()
    {
        return _M_result.end();
    }

    /// See \ref tensor::begin.
    ///
    /// \warning Tensor must be awaited before accessing data with an iterator.
    const_iterator
    begin() const
    {
        return _M_result.begin();
    }

    /// See \ref tensor::end.
    const_iterator
    end() const
    {
        return _M_result.end();
    }

    /// See \ref tensor::expand_dims.
    future_tensor<T, N + 1>
    expand_dims(std::size_t dim) const
    {
        return future_tensor<T, N + 1>(
            _M_result.expand_dims(dim), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    future_tensor<T, M>
    view(int (&&dims)[M]) const requires(M > 0)
    {
        return future_tensor<T, M>(
            _M_result.view(std::move(dims)), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<int, M> dims) const
    {
        return future_tensor<T, M>(
            _M_result.view(dims), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<std::size_t, M> dims) const
    {
        return future_tensor<T, M>(
            _M_result.view(dims), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::flatten.
    template <std::size_t M>
    future_tensor<T, M>
    flatten() const
    {
        return future_tensor<T, M>(
            _M_result.template flatten<M>(), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::narrow.
    future_tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return future_tensor(
            _M_result.narrow(dim, start, length), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::transpose.
    future_tensor
    transpose(const std::size_t (&&dims)[N]) const
    {
        return future_tensor<T, N>(
            _M_result.transpose(std::move(dims)), _M_future_mutex, _M_future, _M_future_wait
        );
    }

    /// See \ref tensor::layout.
    tensor_layout<N>
    layout() const
    {
        return _M_result.layout();
    }

    template <convertible_to_slice... SliceTypes>
    const future_tensor
    operator[](const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return future_tensor(
            _M_result.index_select(slices...), _M_future_mutex, _M_future, _M_future_wait
        );
    }

private:
    future_tensor(
        result_type result,
        future_mutex_type future_mutex,
        future_type future,
        future_wait_type future_wait
    )
    : _M_result(result),
      _M_future_mutex(future_mutex),
      _M_future(future),
      _M_future_wait(future_wait)
    {}

    result_type _M_result;
    future_mutex_type _M_future_mutex;
    future_type _M_future;
    future_wait_type _M_future_wait;

    // Make all specialization of the future tensor friends to the current specialization.
    template <typename FriendT, std::size_t FriendN> friend class future_tensor;
};


/// Deduction guide for the future tensor type.
template <typename T, std::size_t N, asynchronously_invocable Task>
future_tensor(shared_tensor<T, N, hardware_memory_container<T>> t, Task&& task)
    -> future_tensor<T, N>;


template <typename T, std::size_t N>
future_tensor(tensor<T, N, hardware_memory_container<T>>&& t) -> future_tensor<T, N>;


template <immutable_tensor Tensor, hardware_allocator_t<void> Allocator>
future_tensor(Tensor&& t, Allocator accelerator)
    -> future_tensor<typename Tensor::value_type, Tensor::dim()>;


template <typename Tensor, typename T, std::size_t N>
concept is_future_tensor_v = immutable_tensor<Tensor> && std::same_as<Tensor, future_tensor<T, N>>;

template <typename Tensor, typename T>
concept is_future_tensor1_v = is_future_tensor_v<Tensor, T, 1>;

template <typename Tensor, typename T>
concept is_future_tensor2_v = is_future_tensor_v<Tensor, T, 2>;

template <typename Tensor, typename T>
concept is_future_tensor3_v = is_future_tensor_v<Tensor, T, 3>;


template <
    typename T,
    std::size_t N,
    asynchronously_invocable Task,
    hardware_allocator_t<T> Allocator>
auto
empty_future(std::size_t (&&sizes)[N], Task&& task, Allocator alloc)
{
    auto result = shared_tensor(empty<T>(std::move(sizes), alloc));
    auto result_task = task.bind_front(result);

    return future_tensor(result, std::move(result_task));
}


template <
    typename T,
    std::size_t N,
    asynchronously_invocable Task,
    hardware_allocator_t<void> Allocator>
auto
empty_future(std::size_t (&&sizes)[N], Task&& task, Allocator alloc)
{
    return empty_future<T>(
        std::move(sizes), std::move(task), rebind_hardware_allocator<T, Allocator>(alloc)
    );
}


} // namespace metalchat

#pragma once

#include <future>
#include <mutex>

#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_shared.h>


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

    using value_type = result_type::value_type;

    using pointer_type = result_type::pointer_type;

    using container_type = result_type::container_type;

    using iterator = result_type::iterator;

    using const_iterator = result_type::const_iterator;

    template <asynchronously_invocable Task>
    future_tensor(result_type result, Task&& task)
    : _m_result(result),
      _m_future_mutex(std::make_shared<std::mutex>())
    {
        const std::scoped_lock __lock(*_m_future_mutex);

        // Enqueue the calculations to compute the tensor, upon the completion,
        // command buffers calls a callback that sets a value to the promise, so
        // all waiting routines will be unblocked.
        //
        // The main advantage of this approach is that the task and all its associated
        // memory will be released as a result of calling this callback.
        auto future = task([ft = std::make_shared<future_tensor<T, N>>(*this)] {
            const std::scoped_lock __lock(*(ft->_m_future_mutex));

            ft->_m_future_wait = nullptr;
            ft->_m_future = nullptr;
        });

        _m_future = std::make_shared<future_type::element_type>(
            std::bind(&std::shared_future<void>::wait, std::move(future))
        );

        // Erase type of the task, and simply ensure that task is ready, when a user
        // calls either `wait` or `get` method of the future tensor.
        _m_future_wait = std::make_shared<future_wait_type::element_type>(
            std::bind(&Task::make_ready_at_thread_exit, std::move(task))
        );
    }

    template <typename U, std::size_t M>
    future_tensor(result_type result, future_tensor<U, M> future)
    : _m_result(result),
      _m_future_mutex(std::make_shared<std::mutex>())
    {
        const std::scoped_lock __lock(*future._m_future_mutex);

        // Wait on the same future.
        _m_future = future._m_future;
        _m_future_wait = std::make_shared<future_wait_type::element_type>(
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
    : _m_result(result._m_result),
      _m_future_mutex(std::make_shared<std::mutex>())
    {
        _m_future = nullptr;
        _m_future_wait = std::make_shared<future_wait_type::element_type>(
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
    template <asynchronously_invocable Task>
    future_tensor(future_tensor result, Task&& task)
    : future_tensor(result, future_tensor(result._m_result, std::move(task)))
    {}

    /// A naive future tensor that does not wait for any task and returns result immediately.
    future_tensor(result_type result)
    : _m_result(result),
      _m_future_mutex(std::make_shared<std::mutex>()),
      _m_future(nullptr),
      _m_future_wait(nullptr)
    {}

    future_tensor(result_type::tensor_type&& result)
    : future_tensor(shared_tensor(std::move(result)))
    {}

    result_type
    get()
    {
        wait();
        return _m_result;
    }

    void
    wait()
    {
        const std::scoped_lock __lock(*_m_future_mutex);

        if (_m_future_wait) {
            (*_m_future_wait)();
        }

        if (_m_future) {
            (*_m_future)();
        }

        // Once the waiting of the future completion is done, erase associated
        // with this task, but setting an empty function (lambda), so that the
        // task could be destroyed.
        _m_future = nullptr;
        _m_future_wait = nullptr;
    }

    static constexpr std::size_t
    dim()
    {
        return N;
    }

    std::size_t
    numel() const
    {
        return _m_result.numel();
    }

    container_type&
    container() const
    {
        return _m_result.container();
    }

    pointer_type
    data_ptr() const
    {
        return _m_result.data_ptr();
    }

    std::size_t
    size(std::size_t dim) const
    {
        return _m_result.size(dim);
    }

    const std::span<std::size_t, N>
    sizes() const
    {
        return _m_result.sizes();
    }

    std::size_t
    stride(std::size_t dim) const
    {
        return _m_result.stride(dim);
    }

    const std::span<std::size_t, N>
    strides() const
    {
        return _m_result.strides();
    }

    std::size_t
    offset(std::size_t dim) const
    {
        return _m_result.offset(dim);
    }

    const std::span<std::size_t, N>
    offsets() const
    {
        return _m_result.offsets();
    }

    iterator
    begin()
    {
        return _m_result.begin();
    }

    iterator
    end()
    {
        return _m_result.end();
    }

    const_iterator
    begin() const
    {
        return _m_result.begin();
    }

    const_iterator
    end() const
    {
        return _m_result.end();
    }

    future_tensor<T, N + 1>
    expand_dims(std::size_t dim) const
    {
        return future_tensor<T, N + 1>(
            _m_result.expand_dims(dim), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(int (&&dims)[M]) const requires(M > 0)
    {
        return future_tensor<T, M>(
            _m_result.view(std::move(dims)), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<int, M> dims) const
    {
        return future_tensor<T, M>(
            _m_result.view(dims), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<std::size_t, M> dims) const
    {
        return future_tensor<T, M>(
            _m_result.view(dims), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    template <std::size_t M>
    future_tensor<T, M>
    flatten() const
    {
        return future_tensor<T, M>(
            _m_result.template flatten<M>(), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    future_tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return future_tensor(
            _m_result.narrow(dim, start, length), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    future_tensor
    transpose(const std::size_t (&&dims)[N]) const
    {
        return future_tensor<T, N>(
            _m_result.transpose(std::move(dims)), _m_future_mutex, _m_future, _m_future_wait
        );
    }

    tensor_layout<N>
    layout() const
    {
        return _m_result.layout();
    }

    template <indexing::slice_convertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) == N)
    {
        return future_tensor(
            _m_result.index_select(slices...), _m_future_mutex, _m_future, _m_future_wait
        );
    }

private:
    future_tensor(
        result_type result,
        future_mutex_type future_mutex,
        future_type future,
        future_wait_type future_wait
    )
    : _m_result(result),
      _m_future_mutex(future_mutex),
      _m_future(future),
      _m_future_wait(future_wait)
    {}

    result_type _m_result;
    future_mutex_type _m_future_mutex;
    future_type _m_future;
    future_wait_type _m_future_wait;

    // Make all specialization of the future tensor friends to the current specialization.
    template <typename FriendT, std::size_t FriendN> friend class future_tensor;
};


/// Deduction guide for the future tensor type.
template <typename T, std::size_t N, asynchronously_invocable Task>
future_tensor(shared_tensor<T, N, hardware_memory_container<T>> t, Task&& task)
    -> future_tensor<T, N>;


template <typename T, std::size_t N>
future_tensor(tensor<T, N, hardware_memory_container<T>>&& t) -> future_tensor<T, N>;


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

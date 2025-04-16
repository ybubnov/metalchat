#pragma once

#include <future>

#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T>
concept is_waitable = requires(std::remove_reference_t<T> t) {
    { t.wait() } -> std::same_as<void>;
};


class awaitable {
public:
    virtual void
    wait()
        = 0;

    virtual ~awaitable() {}
};


void
wait_all(const std::vector<std::shared_ptr<awaitable>>& awaitables)
{
    for (const auto a : awaitables) {
        a->wait();
    }
}


template <typename T>
concept asynchronously_invocable = requires(std::remove_reference_t<T> t) {
    { t() } -> std::same_as<std::shared_future<void>>;
    { t.make_ready_at_thread_exit() } -> std::same_as<void>;
};


template <typename T, std::size_t N> class future_tensor : public awaitable {
public:
    using result_type = shared_tensor<T, N, device_ref<T>>;

    using future_type = std::shared_future<void>;

    using future_wait_type = std::function<void()>;

    using value_type = result_type::value_type;

    using pointer_type = result_type::pointer_type;

    using container_type = result_type::container_type;

    using iterator = result_type::iterator;

    using const_iterator = result_type::const_iterator;

    future_tensor(const future_tensor& t) noexcept = default;

    template <asynchronously_invocable Task>
    future_tensor(result_type result, Task&& task)
    : _m_result(std::move(result))
    {
        // Enqueue the calculations to compute the tensor, upon the completion,
        // command buffers calls a callback that sets a value to the promise, so
        // all waiting routines will be unblocked.
        _m_future = task();

        // Erase type of the task, and simply ensure that task is ready, when a user
        // calls either `wait` or `get` method of the future tensor.
        _m_future_wait = std::bind(&Task::make_ready_at_thread_exit, std::move(task));
    }

    result_type
    get()
    {
        wait();
        return _m_result;
    }

    void
    wait() override
    {
        _m_future_wait();
        _m_future.wait();

        // Once the waiting of the future completion is done, erase associated
        // with this task, but setting an empty function (lambda), so that the
        // task could be destroyed.
        _m_future_wait = [] {};
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

    NS::SharedPtr<MTL::Buffer>
    memory_move(MTL::Device* device)
    {
        return _m_result.memory_move(device);
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
    offset(std::size_t dim)
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
        return future_tensor<T, N + 1>(_m_result.expand_dims(dim), _m_future, _m_future_wait);
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(int (&&dims)[M]) const requires(M > 0)
    {
        return future_tensor<T, M>(_m_result.view(std::move(dims)), _m_future, _m_future_wait);
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<int, M> dims) const
    {
        return future_tensor<T, M>(_m_result.view(dims), _m_future, _m_future_wait);
    }

    template <std::size_t M>
    future_tensor<T, M>
    view(const std::span<std::size_t, M> dims) const
    {
        return future_tensor<T, M>(_m_result.view(dims), _m_future, _m_future_wait);
    }

    template <std::size_t M>
    future_tensor<T, M>
    flatten() const
    {
        return future_tensor<T, M>(_m_result.template flatten<M>(), _m_future, _m_future_wait);
    }

    future_tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return future_tensor(_m_result.narrow(dim, start, length), _m_future, _m_future_wait);
    }

    future_tensor
    transpose(const std::size_t (&&dims)[N]) const
    {
        return future_tensor(_m_result.transpose(std::move(dims)), _m_future, _m_future_wait);
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
        return future_tensor(_m_result.index_select(slices...), _m_future, _m_future_wait);
    }

private:
    future_tensor(result_type result, future_type future, future_wait_type future_wait)
    : _m_result(result),
      _m_future(future),
      _m_future_wait(future_wait)
    {}

    result_type _m_result;
    future_type _m_future;
    future_wait_type _m_future_wait;

    // Make all specialization of the future tensor friends to the current specialization.
    template <typename FriendT, std::size_t FriendN> friend class future_tensor;
};


/// Deduction guide for the future tensor type.
template <typename T, std::size_t N, asynchronously_invocable Task>
future_tensor(shared_tensor<T, N, device_ref<T>> t, Task&& task) -> future_tensor<T, N>;


template <typename T, std::size_t N>
std::shared_ptr<future_tensor<T, N>>
make_shared(future_tensor<T, N>&& tensor)
{
    return std::make_shared<future_tensor<T, N>>(std::move(tensor));
}


template <typename T, std::size_t N, asynchronously_invocable Task>
auto
empty_future(std::size_t (&&sizes)[N], Task&& task)
{
    auto result = shared_tensor(empty<T>(std::move(sizes), task.device()));
    auto result_task = task.bind_front(result);

    return future_tensor(result, std::move(result_task));
}


} // namespace metalchat

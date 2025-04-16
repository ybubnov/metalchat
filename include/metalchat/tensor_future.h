#pragma once

#include <future>

#include <metalchat/kernel_base.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T, std::size_t N, is_tensor... Args> class future_tensor {
public:
    using result_type = tensor<T, N, device_ref<T>>;

    using kernel_type = kernel_task<Args...>;

    using promise_type = std::promise<void>;

    future_tensor(future_tensor&& t) noexcept = default;

    future_tensor(shared_tensor<T, N, device_ref<T>>&& t, kernel_task<Args...>&& task)
    : _m_result(std::move(t)),
      _m_kernel_task(std::move(task)),
      _m_promise(std::make_shared<promise_type>())
    {
        // Push the calculations to compute the tensor.
        _m_future = std::shared_future(_m_promise->get_future());
        _m_kernel_task(_m_promise);
    }

    result_type&
    get()
    {
        _m_future.wait();
        return *_m_result;
    }

private:
    shared_tensor<T, N, device_ref<T>> _m_result;
    kernel_type _m_kernel_task;
    std::shared_ptr<promise_type> _m_promise;
    std::shared_future<void> _m_future;
};


template <typename T, std::size_t N, is_tensor... Args>
auto
empty_future(const std::size_t (&&sizes)[N], kernel_task<Args...>&& task)
{
    using Result = shared_tensor<T, N, device_ref<T>>;

    auto result = shared_tensor(empty<T>(std::move(sizes), task.device()));
    auto fn = task.bind_front(result);

    return future_tensor<T, N, Result, Args...>(std::move(result), std::move(fn));
}


} // namespace metalchat

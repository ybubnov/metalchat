#pragma once

#include <format>
#include <future>
#include <ranges>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <metalchat/tensor.h>


namespace metalchat {


struct kernel_traits {
    using pipeline_type = NS::SharedPtr<MTL::ComputePipelineState>;
    using queue_type = NS::SharedPtr<MTL::CommandQueue>;
    using kernel_type = NS::SharedPtr<MTL::Function>;
    using buffer_type = NS::SharedPtr<MTL::Buffer>;
    using command_buffer_type = NS::SharedPtr<MTL::CommandBuffer>;
};


template <typename T, std::size_t N, ContiguousContainer Container>
kernel_traits::buffer_type
make_buffer(device& device, const tensor_base<T, N, Container>& t)
{
    auto size = t.numel() * sizeof(T);
    return NS::TransferPtr(device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared));
}

template <typename T, std::size_t N>
kernel_traits::buffer_type
make_buffer(device& device, const tensor_base<T, N, device_ref<T>>& t)
{
    return t.container().storage();
}

template <typename T>
kernel_traits::buffer_type
make_buffer(device& device, const tensor_base<T, 0, value_ref<T>>& t)
{
    return kernel_traits::buffer_type();
}


template <std::size_t N>
kernel_traits::buffer_type
make_buffer(device& device, const tensor_base<tensor_layout<N>, 0, value_ref<tensor_layout<N>>>& t)
{
    auto size = sizeof(tensor_layout<N>);
    return NS::TransferPtr(device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared));
}


inline std::size_t
ceil_div(std::size_t a, std::size_t b)
{
    return (a + b - 1) / b;
}


class execution_policy {
private:
    kernel_traits::pipeline_type _m_pipeline;
    kernel_traits::queue_type _m_queue;
    device& _m_device;

    const dim3 _m_threads;
    const dim3 _m_thread;

public:
    execution_policy(
        kernel_traits::pipeline_type pipeline,
        kernel_traits::queue_type queue,
        device& device,
        const dim3& threads,
        const dim3& thread
    )
    : _m_pipeline(pipeline),
      _m_queue(queue),
      _m_device(device),
      _m_threads(threads),
      _m_thread(thread)
    {
        auto max_threadgroup_size = pipeline->maxTotalThreadsPerThreadgroup();
        if (thread.numel() > max_threadgroup_size) {
            throw std::invalid_argument(std::format(
                "<{}, {}, {}> exceeds maximum number of threads per threadgroup {}", thread.x,
                thread.y, thread.z, max_threadgroup_size
            ));
        }

        if (threads.numel() < thread.numel()) {
            throw std::invalid_argument(std::format(
                "threads per grid <{}, {}, {}> are lesser than threads per group <{}, {}, {}>",
                threads.x, threads.y, threads.z, thread.x, thread.y, thread.z
            ));
        }
    }

    template <typename... T, std::size_t... N, ContiguousContainer... Container>
    kernel_traits::command_buffer_type
    initialize(const tensor_base<T, N, Container>&... args)
    {
        std::size_t i = 0;
        auto command_buf = NS::TransferPtr(_m_queue->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        command_encoder->setComputePipelineState(_m_pipeline.get());
        std::vector<NS::SharedPtr<MTL::Buffer>> buffers;

        ([&] {
            if (auto buf = make_buffer(_m_device, args); buf) {
                buffers.push_back(buf);
                command_encoder->setBuffer(buf.get(), 0, i);
            } else {
                const void* data_ptr = args.data_ptr();
                std::size_t data_size = sizeof(T);
                command_encoder->setBytes(data_ptr, data_size, 0, i);
            }
            i++;
        }(), ...);

        MTL::Size threads_per_grid(_m_threads.x, _m_threads.y, _m_threads.z);
        MTL::Size threads_per_group(_m_thread.x, _m_thread.y, _m_thread.z);
        command_encoder->dispatchThreads(threads_per_grid, threads_per_group);
        command_encoder->endEncoding();

        return command_buf;
    }
};


class sequenced_policy : public execution_policy {
public:
    using execution_policy::execution_policy;

    template <typename... T, std::size_t... N, ContiguousContainer... Container>
    void
    operator()(const tensor_base<T, N, Container>&... args)
    {
        auto command_buf = initialize(args...);
        command_buf->commit();
        command_buf->waitUntilCompleted();
    }
};


template <typename T, std::size_t N> struct unsequenced_policy_callback {
    std::shared_ptr<tensor<T, N, device_ref<T>>> output;
    std::shared_ptr<std::promise<bool>> promise;
};


template <typename T, std::size_t N> class unsequenced_policy : public execution_policy {
private:
    unsequenced_policy_callback<T, N> _m_callback;
    std::future<bool> _m_future;

public:
    unsequenced_policy(
        kernel_traits::pipeline_type pipeline,
        kernel_traits::queue_type queue,
        device& device,
        const dim3& threads,
        const dim3& thread,
        unsequenced_policy_callback<T, N>&& callback
    )
    : execution_policy(pipeline, queue, device, threads, thread),
      _m_callback(std::move(callback))
    {
        _m_future = _m_callback.promise->get_future();
    }

    template <typename... U, std::size_t... M, ContiguousContainer... Container>
    void
    operator()(const tensor_base<U, M, Container>&... args)
    {
        auto command_buf = initialize(args...);
        command_buf->addCompletedHandler([promise
                                          = _m_callback.promise](const MTL::CommandBuffer* buf) {
            promise->set_value(true);
        });
        command_buf->commit();
    }

    tensor<T, N, device_ref<T>>&
    get()
    {
        if (_m_future.get()) {
            return *_m_callback.output;
        }
        throw std::runtime_error("METAL ERROR");
    }
};


class base_kernel {
protected:
    std::string m_op;
    device& m_device;

    kernel_traits::kernel_type m_fn;
    kernel_traits::pipeline_type m_pipeline;
    kernel_traits::queue_type m_queue;

public:
    base_kernel(const std::string& op, device& device)
    : m_op(op),
      m_device(device),
      m_fn(device.make_fn(op))
    {
        NS::Error* error = nullptr;
        m_pipeline = NS::TransferPtr(device->newComputePipelineState(m_fn.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        m_queue = NS::TransferPtr(device->newCommandQueue());
    }

    base_kernel(const std::string& op, const std::string& type, device& device)
    : base_kernel(std::format("{}_{}", op, type), device)
    {}

    std::string
    name() const
    {
        return m_op;
    }

    sequenced_policy
    blocking(dim3 threads, dim3 thread)
    {
        return sequenced_policy(m_pipeline, m_queue, m_device, threads, thread);
    }

    template <typename T, std::size_t N>
    unsequenced_policy<T, N>
    nonblocking(dim3 threads, dim3 thread, unsequenced_policy_callback<T, N>&& cb)
    {
        return unsequenced_policy(m_pipeline, m_queue, m_device, threads, thread, std::move(cb));
    }
};


} // namespace metalchat

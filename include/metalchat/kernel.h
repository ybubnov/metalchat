#pragma once

#include <format>
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
};


template <typename T, std::size_t N, ContiguousContainer Container>
kernel_traits::buffer_type
make_buffer(device& device, const tensor<T, N, Container>& t)
{
    auto size = t.numel() * sizeof(T);
    std::cout << "buffer(ptr)=" << t.data_ptr() << "; [0]=" << t.data_ptr()[0] << std::endl;
    return NS::TransferPtr(device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared));
}

template <typename T, std::size_t N>
kernel_traits::buffer_type
make_buffer(device& device, const tensor<T, N, device_ref<T>>& t)
{
    return t.container().storage();
}

template <typename T>
kernel_traits::buffer_type
make_buffer(device& device, const tensor<T, 0, value_ref<T>>& t)
{
    return kernel_traits::buffer_type();
}


inline std::size_t
ceil_div(std::size_t a, std::size_t b)
{
    return (a + b - 1) / b;
}


class blocking_kernel {
private:
    const std::string m_op;
    device& m_device;
    kernel_traits::pipeline_type m_pipeline;
    kernel_traits::queue_type m_queue;

    const dim3 m_threads;
    const dim3 m_thread;

public:
    blocking_kernel(
        const std::string& op,
        device& device,
        kernel_traits::pipeline_type pipeline,
        kernel_traits::queue_type queue,
        const dim3& threads,
        const dim3& thread
    )
    : m_op(op),
      m_device(device),
      m_pipeline(pipeline),
      m_queue(queue),
      m_threads(threads),
      m_thread(thread)
    {
        auto max_threadgroup_size = pipeline->maxTotalThreadsPerThreadgroup();
        std::cout << "!!!" << threads << " !! " << thread << std::endl;
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
    void
    operator()(const tensor<T, N, Container>&... args)
    {
        std::cout << "blocking_kernel<" << m_threads << ", " << m_thread << ">"
                  << "(" << m_op << ", args[" << sizeof...(args) << "])" << std::endl;
        std::cout << "max threads per threadgroup=" << m_pipeline->maxTotalThreadsPerThreadgroup()
                  << std::endl;

        auto command_buf = NS::TransferPtr(m_queue->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        constexpr auto args_size = sizeof...(args);
        std::array<kernel_traits::buffer_type, args_size> buffers
            = {(make_buffer(m_device, args))...};

        std::array<const void*, args_size> data_ptrs = {(args.data_ptr())...};
        std::array<std::size_t, args_size> data_sizes = {(sizeof(T))...};

        command_encoder->setComputePipelineState(m_pipeline.get());
        for (std::size_t i = 0; i < args_size; i++) {
            if (buffers[i]) {
                command_encoder->setBuffer(buffers[i].get(), 0, i);
            } else {
                command_encoder->setBytes(data_ptrs[i], data_sizes[i], 0, i);
            }
        }

        MTL::Size threads_per_grid(m_threads.x, m_threads.y, m_threads.z);
        MTL::Size threads_per_group(m_thread.x, m_thread.y, m_thread.z);
        command_encoder->dispatchThreads(threads_per_grid, threads_per_group);

        command_encoder->endEncoding();
        command_buf->commit();
        command_buf->waitUntilCompleted();
    }
};


class kernel {
protected:
    std::string m_op;
    device& m_device;

    kernel_traits::kernel_type m_fn;
    kernel_traits::pipeline_type m_pipeline;
    kernel_traits::queue_type m_queue;

public:
    kernel(const std::string& op, device& device)
    : m_op(op),
      m_device(device),
      m_fn(device.make_fn(op))
    {
        std::cout << "init func<" << op << ">" << std::endl;
        NS::Error* error = nullptr;
        m_pipeline = NS::TransferPtr(device->newComputePipelineState(m_fn.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        m_queue = NS::TransferPtr(device->newCommandQueue());
    }

    kernel(const std::string& op, const std::string& type, device& device)
    : kernel(std::format("{}_{}", op, type), device)
    {}

    std::string
    name() const
    {
        return m_op;
    }

    blocking_kernel
    blocking(dim3 threads, dim3 thread)
    {
        return blocking_kernel(m_op, m_device, m_pipeline, m_queue, threads, thread);
    }
};


} // namespace metalchat

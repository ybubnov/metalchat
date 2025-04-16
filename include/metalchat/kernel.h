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


template <typename T, std::size_t N, template <typename U> class Reference>
kernel_traits::buffer_type
make_buffer(device& device, const tensor<T, N, Reference>& t)
{
    auto size = t.numel() * sizeof(T);
    std::cout << "buffer(ptr)=" << t.data_ptr() << "; [0]=" << t.data_ptr()[0] << std::endl;
    return NS::TransferPtr(device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared));
}

template <typename T, std::size_t N>
kernel_traits::buffer_type
make_buffer(device& device, const tensor<T, N, device_ref>& t)
{
    return t.storage()->m_buf;
}


class blocking_kernel {
private:
    const std::string m_op;
    device& m_device;
    kernel_traits::pipeline_type m_pipeline;
    kernel_traits::queue_type m_queue;

    const dim3 m_blocks;
    const dim3 m_threads;

public:
    blocking_kernel(
        const std::string& op,
        device& device,
        kernel_traits::pipeline_type pipeline,
        kernel_traits::queue_type queue,
        const dim3& blocks,
        const dim3& threads
    )
    : m_op(op),
      m_device(device),
      m_pipeline(pipeline),
      m_queue(queue),
      m_blocks(blocks),
      m_threads(threads)
    {
        auto max_threadgroup_size = pipeline->maxTotalThreadsPerThreadgroup();
        if (threads.numel() > max_threadgroup_size) {
            throw std::invalid_argument(std::format(
                "<{}> exceeds maximum number of threads per threadgroup {}", threads.x, threads.y,
                threads.z, max_threadgroup_size
            ));
        }
    }

    template <typename... T, std::size_t... N, template <typename U> class... Reference>
    void
    operator()(const tensor<T, N, Reference>&... args)
    {
        std::cout << "blocking_kernel<" << m_blocks << ", " << m_threads << ">"
                  << "(" << m_op << ", args[" << sizeof...(args) << "])" << std::endl;
        std::cout << "max threads per threadgroup=" << m_pipeline->maxTotalThreadsPerThreadgroup()
                  << std::endl;

        auto command_buf = NS::TransferPtr(m_queue->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        constexpr auto args_size = sizeof...(args);
        std::array<kernel_traits::buffer_type, args_size> buffers
            = {(make_buffer(m_device, args))...};

        command_encoder->setComputePipelineState(m_pipeline.get());
        for (std::size_t i = 0; i < args_size; i++) {
            command_encoder->setBuffer(buffers[i].get(), 0, i);
        }

        MTL::Size grid_blocks(m_blocks.x, m_blocks.y, m_blocks.z);
        MTL::Size grid_threads(m_threads.x, m_threads.y, m_threads.z);
        command_encoder->dispatchThreadgroups(grid_blocks, grid_threads);

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
        NS::Error* error = nullptr;
        m_pipeline = NS::TransferPtr(device->newComputePipelineState(m_fn.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        m_queue = NS::TransferPtr(device->newCommandQueue());
    }

    std::string
    name() const
    {
        return m_op;
    }

    blocking_kernel
    blocking(dim3 blocks, dim3 threads)
    {
        return blocking_kernel(m_op, m_device, m_pipeline, m_queue, blocks, threads);
    }
};


} // namespace metalchat

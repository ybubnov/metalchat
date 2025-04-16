#pragma once

#include <ranges>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


class operation {
protected:
    std::string m_op;
    device& m_device;

    NS::SharedPtr<MTL::Function> m_fn;
    NS::SharedPtr<MTL::ComputePipelineState> m_pipeline;
    NS::SharedPtr<MTL::CommandQueue> m_commandq;

public:
    using buffer_type = NS::SharedPtr<MTL::Buffer>;

    operation(const std::string& op, device& device)
    : m_op(op),
      m_device(device),
      m_fn(device.make_fn(op))
    {
        NS::Error* error = nullptr;
        m_pipeline = NS::TransferPtr(device->newComputePipelineState(m_fn.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        m_commandq = NS::TransferPtr(this->m_device->newCommandQueue());
    }

    std::string
    name() const
    {
        return m_op;
    }

    template <typename... T, std::size_t... N, template <typename U> class... Reference>
    void
    blocking_kernel(dim3 blocks, dim3 threads, const tensor<T, N, Reference>&... args)
    {
        std::cout << "blocking_kernel(" << m_op << ", args[" << sizeof...(args) << "])"
                  << std::endl;
        auto command_buf = NS::TransferPtr(m_commandq->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        constexpr auto args_size = sizeof...(args);
        std::array<buffer_type, args_size> buffers = {(make_device_buffer(args))...};

        command_encoder->setComputePipelineState(m_pipeline.get());
        for (std::size_t i = 0; i < args_size; i++) {
            command_encoder->setBuffer(buffers[i].get(), 0, i);
        }

        MTL::Size grid_blocks(blocks.x, blocks.y, blocks.z);
        MTL::Size grid_threads(threads.x, threads.y, threads.y);
        command_encoder->dispatchThreadgroups(grid_blocks, grid_threads);

        command_encoder->endEncoding();
        command_buf->commit();
        command_buf->waitUntilCompleted();
    }

    template <typename T, std::size_t N, template <typename U> class Reference>
    buffer_type
    make_device_buffer(const tensor<T, N, Reference>& t)
    {
        auto size = t.numel() * sizeof(T);
        std::cout << "buffer(ptr)=" << t.data_ptr() << "; [0]=" << t.data_ptr()[0] << std::endl;
        return NS::TransferPtr(
            m_device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared)
        );
    }

    template <typename T, std::size_t N>
    buffer_type
    make_device_buffer(const tensor<T, N, device_ref>& t)
    {
        return t.storage()->m_buf;
    }
};


} // namespace nn
} // namespace metalchat

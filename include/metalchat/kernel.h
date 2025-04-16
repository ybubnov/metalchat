#pragma once


#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <format>
#include <future>
#include <tuple>

#include <metalchat/container.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


struct dim3 {
    const std::size_t x, y, z;

    constexpr dim3(std::size_t x_, std::size_t y_ = 1, std::size_t z_ = 1)
    : x(x_),
      y(y_),
      z(z_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const dim3& d)
    {
        os << "<" << d.x << "," << d.y << "," << d.z << ">";
        return os;
    }

    std::size_t
    numel() const
    {
        return x * y * z;
    }
};


class kernel_base {
private:
    NS::SharedPtr<MTL::Function> _m_function;
    NS::SharedPtr<MTL::ComputePipelineState> _m_pipeline;
    NS::SharedPtr<MTL::CommandQueue> _m_queue;
    NS::SharedPtr<MTL::Device> _m_device;

    static NS::SharedPtr<MTL::Function>
    _m_initialize_function(const std::string& name, NS::SharedPtr<MTL::Library> library)
    {
        auto fn_name = NS::TransferPtr(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        return NS::TransferPtr(library->newFunction(fn_name.get()));
    }

public:
    kernel_base(const kernel_base& k) noexcept = default;

    kernel_base(
        const std::string& name,
        NS::SharedPtr<MTL::Device> device,
        NS::SharedPtr<MTL::Library> library
    )
    : _m_function(_m_initialize_function(name, library)),
      _m_pipeline(),
      _m_queue(),
      _m_device(device)
    {
        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        _m_pipeline
            = NS::TransferPtr(device->newComputePipelineState(_m_function.get(), &error_ptr));
        if (!_m_pipeline) {
            throw std::runtime_error(std::format(
                "base_kernel: failed to create compute pipeline, {}",
                error_ptr->localizedDescription()->utf8String()
            ));
        }

        _m_queue = NS::TransferPtr(device->newCommandQueue());
    }

    std::size_t
    max_threads_per_threadgroup()
    {
        return _m_pipeline->maxTotalThreadsPerThreadgroup();
    }

    MTL::Device*
    device()
    {
        return _m_device.get();
    }

    MTL::ComputePipelineState*
    pipeline()
    {
        return _m_pipeline.get();
    }

    MTL::CommandBuffer*
    make_buffer()
    {
        return _m_queue->commandBuffer();
    }
};


} // namespace metalchat

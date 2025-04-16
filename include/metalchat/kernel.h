#pragma once

#include <format>
#include <future>
#include <tuple>

#include <metalchat/container.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


class kernel_base {
private:
    std::string _m_name;
    NS::SharedPtr<MTL::Function> _m_function;
    NS::SharedPtr<MTL::ComputePipelineState> _m_pipeline;

    // TODO: is it a good idea to use a reference wrapper here?
    std::reference_wrapper<shared_kernel_thread> _m_kernel_thread;

public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

    kernel_base(
        const std::string& name,
        NS::SharedPtr<MTL::Library> library,
        shared_kernel_thread& kernel_thread
    )
    : _m_name(name),
      _m_function(),
      _m_pipeline(),
      _m_kernel_thread(kernel_thread)
    {
        auto fn_name = NS::TransferPtr(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        _m_function = NS::TransferPtr(library->newFunction(fn_name.get()));
        if (!_m_function) {
            throw std::invalid_argument(
                std::format("base_kernel: function {} not found in a shader library", name)
            );
        }

        _m_function->setLabel(fn_name.get());

        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        auto device_ptr = library->device();

        auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc());
        descriptor->init();
        descriptor->setComputeFunction(_m_function.get());
        descriptor->setLabel(fn_name.get());

        _m_pipeline = NS::TransferPtr(device_ptr->newComputePipelineState(
            descriptor.get(), MTL::PipelineOptionNone, nullptr, &error_ptr
        ));

        if (!_m_pipeline) {
            throw std::runtime_error(std::format(
                "base_kernel: failed to create compute pipeline, {}",
                error_ptr->localizedDescription()->utf8String()
            ));
        }
    }

    std::string
    name() const
    {
        return _m_name;
    }

    allocator_type
    get_allocator()
    {
        return _m_kernel_thread.get().get_allocator();
    }

    std::size_t
    max_threads_per_threadgroup()
    {
        return _m_pipeline->maxTotalThreadsPerThreadgroup();
    }

    std::shared_ptr<kernel_thread>
    get_this_thread()
    {
        return _m_kernel_thread.get().get_this_thread();
    }

    MTL::ComputePipelineState*
    pipeline()
    {
        return _m_pipeline.get();
    }
};


} // namespace metalchat

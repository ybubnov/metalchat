#pragma once

#include <filesystem>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


namespace metalchat {
namespace metal {


struct buffer {
    NS::SharedPtr<MTL::Buffer> ptr;

    buffer(NS::SharedPtr<MTL::Buffer> p)
    : ptr(p)
    {}

    buffer(MTL::Buffer* p)
    : ptr(NS::TransferPtr(p))
    {}
};


shared_buffer
make_buffer(NS::SharedPtr<MTL::Buffer> p);


shared_buffer
make_buffer(MTL::Buffer* p);


struct device {
    NS::SharedPtr<MTL::Device> ptr;

    device(NS::SharedPtr<MTL::Device> p)
    : ptr(p)
    {}

    device(MTL::Device* p)
    : ptr(NS::TransferPtr(p))
    {}
};


shared_device
make_device();


struct kernel {
    NS::SharedPtr<MTL::Function> function;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline;

    kernel(NS::SharedPtr<MTL::Function> f, NS::SharedPtr<MTL::ComputePipelineState> p)
    : function(f),
      pipeline(p)
    {}

    kernel(MTL::Function* f, MTL::ComputePipelineState* p)
    : function(NS::TransferPtr(f)),
      pipeline(NS::TransferPtr(p))
    {}
};


struct library {
    NS::SharedPtr<MTL::Library> ptr;

    library(NS::SharedPtr<MTL::Library> p)
    : ptr(p)
    {}

    library(MTL::Library* p)
    : ptr(NS::TransferPtr(p))
    {}
};


shared_library
make_library(const NS::URL* url, shared_device device);


shared_library
make_library(const std::filesystem::path& p, shared_device device);


} // namespace metal
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <list>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>


namespace metalchat {
namespace metal {


struct buffer {
    using deleter_type = std::function<void(buffer* p)>;

    MTL::Buffer* ptr;

    buffer(MTL::Buffer* p)
    : ptr(p)
    {}
};


struct buffer_deleter {
    std::list<buffer::deleter_type> deleters;

    buffer_deleter()
    : deleters()
    {}

    buffer_deleter(buffer::deleter_type deleter)
    : deleters({deleter})
    {}

    void
    invoke_before_destroy(buffer::deleter_type deleter)
    {
        deleters.push_back(deleter);
    }

    void
    operator()(buffer* b)
    {
        for (auto& deleter : deleters) {
            deleter(b);
        }

        b->ptr->release();
        b->ptr = nullptr;
        delete b;
    }
};


shared_buffer
make_buffer(MTL::Buffer* p);


shared_buffer
make_buffer(MTL::Buffer* p, buffer::deleter_type deleter);


struct device {
    NS::SharedPtr<MTL::Device> ptr;

    device(NS::SharedPtr<MTL::Device> p)
    : ptr(p)
    {}

    device(MTL::Device* p)
    : ptr(NS::RetainPtr(p))
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
    : function(NS::RetainPtr(f)),
      pipeline(NS::RetainPtr(p))
    {}
};


struct library {
    NS::SharedPtr<MTL::Library> ptr;

    library(NS::SharedPtr<MTL::Library> p)
    : ptr(p)
    {}

    library(MTL::Library* p)
    : ptr(NS::RetainPtr(p))
    {}
};


shared_library
make_library(const NS::URL* url, shared_device device);


shared_library
make_library(const std::filesystem::path& p, shared_device device);


} // namespace metal
} // namespace metalchat

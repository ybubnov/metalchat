#pragma once


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


} // namespace metal
} // namespace metalchat

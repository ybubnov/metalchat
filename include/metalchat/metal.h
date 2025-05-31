#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


namespace metalchat {
namespace metal {


class buffer {
public:
    struct impl {
        NS::SharedPtr<MTL::Buffer> buf;
    };

    buffer(impl&&);

    /// Returns a raw pointer to the underlying buffer serving as element storage.
    void*
    data();
    void*
    data() const;

    MTL::Buffer*
    get()
    {
        return _m_impl->buf.get();
    }

private:
    std::shared_ptr<impl> _m_impl;
};


struct device;

using shared_device = std::shared_ptr<device>;


} // namespace metal
} // namespace metalchat

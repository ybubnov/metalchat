#pragma once

#include "metal_impl.h"


namespace metalchat {
namespace metal {


struct device {
    NS::SharedPtr<MTL::Device> ptr;

    device(NS::SharedPtr<MTL::Device> p)
    : ptr(p)
    {}
};


} // namespace metal
} // namespace metalchat

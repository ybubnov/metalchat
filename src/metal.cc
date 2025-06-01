#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <metalchat/metal.h>

#include "metal_impl.h"


namespace metalchat {
namespace metal {


void*
data(const shared_buffer buffer)
{
    return buffer->ptr->contents();
}


shared_buffer
make_buffer(NS::SharedPtr<MTL::Buffer> p)
{
    return std::make_shared<buffer>(p);
}


shared_buffer
make_buffer(MTL::Buffer* p)
{
    return std::make_shared<buffer>(p);
}


} // namespace metal
} // namespace metalchat

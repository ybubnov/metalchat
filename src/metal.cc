#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <metalchat/metal.h>


namespace metalchat {
namespace metal {


// struct buffer::impl {
//     NS::SharedPtr<MTL::Buffer> buf;
// };


buffer::buffer(buffer::impl&& buffer_impl)
: _m_impl(std::make_shared<impl>(buffer_impl))
{}


void*
buffer::data()
{
    return _m_impl->buf->contents();
}


void*
buffer::data() const
{
    return _m_impl->buf->contents();
}


} // namespace metal
} // namespace metalchat

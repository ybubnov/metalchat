// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <metalchat/metal.h>

#include "metal_impl.h"


namespace metalchat {
namespace metal {


void*
data(const shared_buffer buffer)
{
    return buffer->ptr->contents();
}


std::size_t
size(const shared_buffer buffer)
{
    return buffer->ptr->length();
}


shared_buffer
make_buffer(MTL::Buffer* p)
{
    auto buffer_ptr = new buffer(p);
    return shared_buffer(buffer_ptr, buffer_deleter());
}


shared_buffer
make_buffer(MTL::Buffer* p, buffer::deleter_type deleter)
{
    auto buffer_ptr = new buffer(p);
    return shared_buffer(buffer_ptr, buffer_deleter(deleter));
}


shared_device
make_device()
{
    return std::make_shared<device>(MTL::CreateSystemDefaultDevice());
}


shared_library
make_library(const NS::URL* url, shared_device device)
{
    NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
    NS::Error* error_ptr = error.get();

    auto library_ptr = NS::TransferPtr(device->ptr->newLibrary(url, &error_ptr));
    if (!library_ptr) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(failure_reason->utf8String());
    }

    return std::make_shared<library>(library_ptr);
}


shared_library
make_library(const std::filesystem::path& path, shared_device device)
{
    auto path_str = path.string();
    auto path_cstr = path_str.c_str();

    auto filepath = NS::TransferPtr(NS::String::string(path_cstr, NS::UTF8StringEncoding));
    auto url_ptr = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));

    return make_library(url_ptr.get(), device);
}


} // namespace metal
} // namespace metalchat

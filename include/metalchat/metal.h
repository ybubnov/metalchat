#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


namespace metalchat {
namespace metal {


struct buffer;
using shared_buffer = std::shared_ptr<buffer>;

void*
data(const shared_buffer buffer);


struct device;
using shared_device = std::shared_ptr<device>;


struct kernel;
using shared_kernel = std::shared_ptr<kernel>;


struct library;
using shared_library = std::shared_ptr<library>;


} // namespace metal
} // namespace metalchat

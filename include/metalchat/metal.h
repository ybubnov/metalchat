// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>


namespace metalchat {
namespace metal {


struct buffer;
using shared_buffer = std::shared_ptr<buffer>;

void*
data(const shared_buffer buffer);


std::size_t
size(const shared_buffer buffer);


struct device;
using shared_device = std::shared_ptr<device>;


struct kernel;
using shared_kernel = std::shared_ptr<kernel>;


struct library;
using shared_library = std::shared_ptr<library>;


} // namespace metal
} // namespace metalchat

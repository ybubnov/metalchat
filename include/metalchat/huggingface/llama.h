// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/autoloader.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace huggingface {


struct metallama3_document_adaptor {
    safetensor_document
    adapt(const safetensor_document& document) const;
};


template <typename T = bf16, contiguous_container Container = hardware_memory_container<T>>
struct metallama3_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using options_type = nn::llama3_options;
    using container_type = Container;

    using document_adaptor = metallama3_document_adaptor;
};


using metallama3_autoloader = autoloader<metallama3_traits<bf16>>;


} // namespace huggingface
} // namespace metalchat

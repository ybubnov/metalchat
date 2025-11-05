// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <mstch/mstch.hpp>

#include <metalchat/interpreter.h>

#include "metal_impl.h"


namespace mustache = mstch;


namespace metalchat {


void
interpreter::write_header(const std::string& role)
{
    auto output = std::back_inserter(_M_buf);

    _M_encoder.encode(text::special_token::begin_header, output);
    _M_encoder.encode(role, output);
    _M_encoder.encode(text::special_token::end_header, output);
    _M_encoder.encode("\n\n", output);
}


void
interpreter::write(const basic_message& message)
{
    write_header(message.role());

    auto output = std::back_inserter(_M_buf);
    auto content = mustache::render(message.content(), mustache::node());
    _M_encoder.encode(content, output);
    _M_encoder.encode(text::special_token::end_turn, output);
}


interpreter
make_llama3(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<nn::llama3_options> options_
)
{
    metalchat::text::bpe bpe(tokens_path);
    metalchat::hardware_accelerator gpu0;

    auto options = options_.value_or(nn::default_llama3_1b_options());

    using container_type = hardware_memory_container<bf16>;
    using transformer_type = nn::llama3<bf16, container_type>;

    auto transformer = transformer_type(options, gpu0);
    safetensor_document::load(weights_path, transformer);

    auto device = gpu0.get_metal_device();

    if (options.heap_size() > 0) {
        auto alloc = hardware_heap_allocator<void>(device, options.heap_size());
        gpu0.set_allocator(nocopy_allocator(alloc, device));
    }

    return interpreter(std::move(transformer), bpe);
}


/*
interpreter
make_llama3_compact(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<nn::llama3_options> options_
)
{
    metalchat::bpe bpe(tokens_path);
    metalchat::hardware_accelerator gpu0(64);

    auto alloc0 = hardware_resident_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc0));

    auto weights_file = std::make_shared<basic_memfile>(weights_path);

    auto options = options_.value_or(nn::default_llama3_1b_options());

    using value_type = dtype::bf16;
    using container_type = filebuf_memory_container<value_type>;
    using transformer_type = nn::llama3<value_type, container_type>;

    auto transformer = transformer_type(options, gpu0);

    auto alloc = filebuf_memory_allocator<value_type>();
    auto weights = safetensor_document(weights_file, safetensor_openmode::in);
    weights.load(transformer, alloc);

    auto alloc3 = hardware_heap_allocator<void>(gpu0.get_metal_device(), options.heap_size());
    auto alloc4 = hardware_nocopy_allocator(alloc3, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc4));

    return interpreter(std::move(transformer), bpe);
}
*/


} // namespace metalchat

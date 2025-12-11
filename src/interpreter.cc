// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <mstch/mstch.hpp>

#include <metalchat/autoloader.h>
#include <metalchat/interpreter.h>

#include "metal_impl.h"


namespace mustache = mstch;


namespace metalchat {


const std::string interpreter::variable::commands = "METALCHAT_COMMANDS";
const std::string interpreter::variable::command_format = "METALCHAT_COMMAND_FORMAT";


namespace variable_default {


static const std::string command_format = R"(
To use a tool, respond with JSON in this format:
{"name":"command_name","parameters":{"param1":"value","param2":"value"}}
)";


} // namespace variable_default


struct interpreter::_Members {
    mustache::map vars;

    _Members()
    : vars({
          {expand(variable::commands), ""},
          {expand(variable::command_format), variable_default::command_format},
      })
    {}

    std::string
    expand(const std::string& key)
    {
        return "$" + key;
    }

    std::string&
    at(const std::string& key)
    {
        return std::get<std::string>(vars[expand(key)]);
    }

    void
    assign(const std::string& key, const std::string& val)
    {
        vars.insert_or_assign(expand(key), val);
    }
};


interpreter::interpreter(
    std::shared_ptr<basic_transformer> transformer_ptr,
    const text::bpe& encoder,
    std::size_t max_pos
)
: _M_members(std::make_shared<_Members>()),
  _M_transformer(transformer_ptr),
  _M_command_scanner(std::make_shared<json_command_scanner>()),
  _M_commands(),
  _M_encoder(encoder),
  _M_max_pos(max_pos),
  _M_start_pos(0),
  _M_buf(1, encoder.encode(text::special_token::begin_text))
{
    // Do not escape characters, leave them as is. This is the global configuration,
    // so unfortunately this line changes behaviour for the whole library.
    mustache::config::escape = [](const std::string& str) -> std::string { return str; };
}


void
interpreter::declare_command(const std::string& declaration, command_type command)
{
    auto command_name = _M_command_scanner->declare(declaration);
    _M_commands.insert_or_assign(command_name, command);

    auto& commands = _M_members->at(variable::commands);
    if (!commands.empty()) {
        commands += ",";
    }
    commands += declaration;
}


void
interpreter::declare_variable(const std::string& declaration, const std::string& value)
{
    if (declaration.starts_with("$")) {
        throw std::invalid_argument(
            std::format("interpreter: variable {} cannot start with $", declaration)
        );
    }

    _M_members->assign(declaration, value);
}


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
    auto content = mustache::render(message.content(), _M_members->vars);
    _M_encoder.encode(content, output);
    _M_encoder.encode(text::special_token::end_turn, output);
}


interpreter
make_llama3(const std::filesystem::path& weights_path, const std::filesystem::path& tokens_path)
{
    hardware_accelerator accelerator;
    text::bpe bpe(tokens_path);

    using LLama3 = nn::llama3<bf16>;
    nn::indirect_layer<LLama3> layer(nn::default_llama3_1b_options(), accelerator);

    auto document = safetensor_document::open(weights_path, accelerator);
    auto document_adaptor = llama3_reference_traits::document_adaptor();
    document_adaptor.adapt(document);
    document.load(layer);

    struct llama3 : public basic_transformer {
        nn::indirect_layer<LLama3> _M_layer;

        llama3(nn::indirect_layer<LLama3> l)
        : _M_layer(l)
        {}

        tensor_type
        transform(tensor_type input, std::size_t start_pos)
        {
            return _M_layer->transform(input, start_pos);
        }

        hardware_accelerator&
        accelerator()
        {
            return _M_layer.accelerator();
        }
    };

    std::shared_ptr<basic_transformer> layer_ptr = std::make_shared<llama3>(layer);
    return interpreter(layer_ptr, bpe);
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

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cstdlib>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>

#include "config.h"
#include "model.h"
#include "program.h"


namespace metalchat {
namespace runtime {


namespace jsonpath = jsoncons::jsonpath;


program::program()
: basic_command("metalchat"),
  _M_credential(*this),
  _M_model(*this),
  _M_options(*this),
  _M_stdin("-")
{
    auto config_path = std::filesystem::path("~") / default_path / default_config_path;

    _M_command.add_description("A self-sufficient runtime for large language models");
    _M_command.add_argument("-f", "--file")
        .help("read configuration file only from this location")
        .metavar("<config-file>")
        .default_value(config_path.string())
        .nargs(1);

    _M_stdin.add_description("read from stdin and run default model");
    _M_stdin.add_argument("model_id")
        .help("the model to launch for the input processing")
        .required()
        .store_into(_M_model_id);
    push_handler(_M_stdin, [&](const command_context& c) { handle_stdin(c); });
}


void
program::handle_stdin(const command_context& c)
{
    using Transformer = huggingface::llama3;

    model_provider models(c.root_path);
    auto model = models.find(_M_model_id);
    auto repository = filesystem_repository<Transformer>(model.path);

    auto tokenizer = repository.retrieve_tokenizer();
    auto options = repository.retrieve_options();

    if (model.manifest.options) {
        Transformer::options_saver options_saver;
        Transformer::options_loader options_loader;

        std::stringstream input_stream;
        options_saver.save(input_stream, options);

        auto options_doc = jsoncons::json::parse(input_stream);
        for (const auto& [k, option] : model.manifest.options.value()) {
            auto query = std::string("$.") + k;

            std::visit([&](auto&& value) {
                jsonpath::json_replace(options_doc, query, std::move(value));
            }, option);
        }

        std::stringstream output_stream;
        output_stream << options_doc;
        options = options_loader.load(output_stream);
    }

    auto transformer = repository.retrieve_transformer(options);
    auto interp = metalchat::interpreter(transformer, tokenizer);

    const std::size_t max_input_size = 1024;
    std::string input(max_input_size, '\0');

    if (std::cin.read(input.data(), max_input_size)) {
        throw std::runtime_error("failed reading from stdin");
    }
    input = std::string(input.c_str(), std::cin.gcount());

    interp.write(basic_message("system", "You are a helpful assistant"));
    interp.write(basic_message("user", input));

    // TODO: ensure that encoded context does not exceed the model limit.
    std::ostream_iterator<std::string> content_iterator(std::cout << std::unitbuf);
    interp.read(content_iterator);
}


void
program::handle(int argc, char** argv)
{
    _M_command.parse_args(argc, argv);
    auto config_path = _M_command.get<std::string>("--file");
    if (config_path.starts_with("~/")) {
        config_path = std::string(std::getenv("HOME")) + config_path.substr(1);
    }

    auto root_path = std::filesystem::path(config_path).parent_path();
    std::filesystem::create_directories(root_path);

    command_context context{.root_path = root_path, .config_file = tomlfile<config>(config_path)};
    basic_command::handle(context);
}


} // namespace runtime
} // namespace metalchat

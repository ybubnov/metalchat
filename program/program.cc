// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cstdlib>

#include "config.h"
#include "program.h"


namespace metalchat {
namespace runtime {


program::program()
: basic_command("metalchat"),
  _M_credential(*this),
  _M_model(*this),
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
    auto repo_path = c.root_path / "models" / _M_model_id;
    auto repository = filesystem_repository<huggingface::llama3>(repo_path);

    auto tokenizer = repository.retrieve_tokenizer();
    auto transformer = repository.retrieve_transformer();
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

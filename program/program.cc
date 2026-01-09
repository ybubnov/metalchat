// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cstdlib>

#include "config.h"
#include "program.h"


namespace metalchat {
namespace program {


program::program()
: basic_command("metalchat"),
  _M_credential(*this),
  _M_model(*this)
{
    auto config_path = std::filesystem::path("~") / default_path / default_config_path;

    _M_command.add_description("A self-sufficient runtime for large language models");
    _M_command.add_argument("-f", "--file")
        .help("read configuration file only from this location")
        .metavar("<config-file>")
        .default_value(config_path.string())
        .nargs(1);
}


void
program::handle(int argc, char** argv)
{
    _M_command.parse_args(argc, argv);
    auto config_path = _M_command.get<std::string>("--file");
    if (config_path.starts_with("~/")) {
        config_path = std::string(std::getenv("HOME")) + config_path.substr(1);
    }

    auto parent_path = std::filesystem::path(config_path).parent_path();
    if (!std::filesystem::exists(parent_path)) {
        std::filesystem::create_directories(parent_path);
    }

    command_context context{.config_file = tomlfile<config>(config_path)};

    basic_command::handle(context);
}


} // namespace program
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "model.h"
#include "http.h"


namespace metalchat {
namespace program {


std::string architecture::llama3x2_1b = "llama3.2:1b";
std::string architecture::llama3x2_3b = "llama3.2:3b";


model_command::model_command(basic_command& parent)
: basic_command("model", parent),
  _M_pull("pull"),
  _M_list("list"),
  _M_remove("remove"),
  _M_repository(),
  _M_arch()
{
    _M_command.add_description("manage language models");

    _M_pull.add_description("download a model from a remote server");
    _M_pull.add_argument("repository")
        .help("the repository to pull the image from")
        .required()
        .store_into(_M_repository);
    _M_pull.add_argument("name").help("the name of the model").required().store_into(_M_name);
    _M_pull.add_argument("-a", "--arch")
        .help("a model architecture")
        .choices(architecture::llama3x2_1b, architecture::llama3x2_3b)
        .default_value(architecture::llama3x2_1b)
        .nargs(1)
        .store_into(_M_arch);
    _M_list.add_description("list the available models");
    _M_remove.add_description("remove matching models");

    push_handler(_M_pull, [&](const command_context& c) { pull(c); });
    push_handler(_M_list, [&](const command_context& c) { list(c); });
    push_handler(_M_remove, [&](const command_context& c) { remove(c); });
}


void
model_command::pull(const command_context& context)
{
    std::ofstream local_file("index.html", std::ios::binary | std::ios::trunc);
    httpfile remote_file("http://localhost:8000/index.html");

    std::ostream_iterator<char> output(local_file);
    remote_file.read(output);
}


void
model_command::list(const command_context& context)
{}


void
model_command::remove(const command_context& context)
{}


} // namespace program
} // namespace metalchat

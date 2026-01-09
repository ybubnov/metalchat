// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "model.h"
#include "http.h"


namespace metalchat {
namespace program {


std::string architecture::llama3_2 = "llama3.2";

std::string variant::huggingface = "huggingface";

std::string partitioning::sharded = "sharded";
std::string partitioning::consolidated = "consolidated";


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
        .metavar("<architecture>")
        .choices(architecture::llama3_2)
        .default_value(architecture::llama3_2)
        .nargs(1)
        .store_into(_M_arch);
    _M_pull.add_argument("-p", "--partitioning")
        .help("a model partitioning strategy")
        .metavar("<partitioning>")
        .choices(partitioning::consolidated)
        .default_value(partitioning::consolidated)
        .nargs(1)
        .store_into(_M_partitioning);

    _M_list.add_description("list the available models");

    _M_remove.add_description("remove matching models");
    _M_remove.add_argument("name").help("the name of the model").required().store_into(_M_name);

    push_handler(_M_pull, [&](const command_context& c) { pull(c); });
    push_handler(_M_list, [&](const command_context& c) { list(c); });
    push_handler(_M_remove, [&](const command_context& c) { remove(c); });
}


void
model_command::pull(const command_context& context)
{
    // std::ofstream local_file("index.html", std::ios::binary | std::ios::trunc);
    // httpfile remote_file("http://localhost:8000/index.html");

    // std::ostream_iterator<char> output(local_file);
    // remote_file.read(output);

    auto root_path = context.config_file.path().parent_path();
    auto repo_path = root_path / default_path / _M_name;
    auto manifest_path = repo_path / "manifest.toml";

    if (std::filesystem::exists(repo_path)) {
        throw std::invalid_argument(std::format("pull: model '{}' already exists", _M_name));
    }

    manifest m = {
        .model =
            {.variant = "huggingface",
             .repository = _M_repository,
             .architecture = _M_arch,
             .partitioning = _M_partitioning}
    };

    std::filesystem::create_directories(repo_path);

    tomlfile<manifest> manifest_file(manifest_path, tomlformat::multiline);
    manifest_file.write(m);
}


void
model_command::list(const command_context& context)
{}


void
model_command::remove(const command_context& context)
{}


} // namespace program
} // namespace metalchat

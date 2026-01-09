// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "model.h"
#include "http.h"


namespace metalchat {
namespace workspace {


std::string architecture::llama3 = "llama3";
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
        .choices(architecture::llama3)
        .default_value(architecture::llama3)
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

    auto repo_path = context.root_path / default_path / _M_name;
    auto manifest_path = repo_path / manifest_name;

    url u(_M_repository);
    std::cout << "Pulling from '" << u.string() << "'..." << std::endl;

    if (std::filesystem::exists(repo_path)) {
        throw std::invalid_argument(std::format("pull: model '{}' already exists", _M_name));
    }

    manifest manifest_document = {
        .model =
            {.variant = variant::huggingface,
             .repository = _M_repository,
             .architecture = _M_arch,
             .partitioning = _M_partitioning}
    };

    std::filesystem::create_directories(repo_path);

    tomlfile<manifest> manifest_file(manifest_path, tomlformat::multiline);
    manifest_file.write(manifest_document);
}


void
model_command::list(const command_context& context)
{
    auto root_path = context.root_path / default_path;

    for (auto const& repo_path : std::filesystem::directory_iterator(root_path)) {
        if (!std::filesystem::is_directory(repo_path)) {
            continue;
        }

        auto manifest_path = repo_path.path() / manifest_name;
        auto manifest_document = tomlfile<manifest>::read(repo_path / manifest_path);

        std::cout << repo_path.path().filename().string() << '\t';
        std::cout << manifest_document.model.architecture << '\t';
        std::cout << manifest_document.model.partitioning << '\t';
        std::cout << manifest_document.model.repository << std::endl;
    }
}


void
model_command::remove(const command_context& context)
{
    auto repo_path = context.root_path / default_path / _M_name;
    std::filesystem::remove_all(repo_path);
}


} // namespace workspace
} // namespace metalchat

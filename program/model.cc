// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/huggingface.h>
#include <metalchat/repository.h>

#include "credential.h"
#include "digest.h"
#include "http.h"
#include "model.h"


namespace metalchat {
namespace runtime {


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
    _M_pull.add_argument("-a", "--arch")
        .help("a model architecture")
        .metavar("<architecture>")
        .choices(architecture::llama3)
        .default_value(architecture::llama3)
        .nargs(1)
        .store_into(_M_arch);
    _M_pull.add_argument("-V", "--variant")
        .help("a model implementation variant")
        .metavar("<variant>")
        .choices(variant::huggingface)
        .default_value(variant::huggingface)
        .nargs(1)
        .store_into(_M_variant);
    _M_pull.add_argument("-p", "--partitioning")
        .help("a model partitioning strategy")
        .metavar("<partitioning>")
        .choices(partitioning::consolidated)
        .default_value(partitioning::consolidated)
        .nargs(1)
        .store_into(_M_partitioning);

    _M_list.add_description("list the available models");

    _M_remove.add_description("remove matching models");
    _M_remove.add_argument("id").help("a model identifier").required().store_into(_M_id);

    push_handler(_M_pull, [&](const command_context& c) { pull(c); });
    push_handler(_M_list, [&](const command_context& c) { list(c); });
    push_handler(_M_remove, [&](const command_context& c) { remove(c); });
}


void
model_command::pull(const command_context& context)
{
    url u(_M_repository);
    u.push_query("architecture", _M_arch);
    u.push_query("partitioning", _M_partitioning);
    u.push_query("variant", _M_variant);

    auto model_id = sha1(u);
    auto repo_path = context.root_path / default_path / model_id;
    auto manifest_path = repo_path / manifest_name;

    if (std::filesystem::exists(repo_path)) {
        throw std::invalid_argument("pull: model already exists");
    }

    manifest manifest_document = {
        .model =
            {.variant = _M_variant,
             .repository = _M_repository,
             .architecture = _M_arch,
             .partitioning = _M_partitioning}
    };

    std::cout << "Pulling from '" << _M_repository << "'..." << std::endl;

    using filesystem_type = http_tracking_filesystem;
    using transformer_type = huggingface::llama3;
    using http_repository = huggingface_repository<transformer_type, filesystem_type>;

    http_bearer_auth<keychain_provider> http_auth;
    http_tracking_filesystem filesystem(url(_M_repository), http_auth);
    http_repository repository(u.path(), repo_path, filesystem);

    repository.clone();

    tomlfile<manifest> manifest_file(manifest_path, tomlformat::multiline);
    manifest_file.write(manifest_document);
}


void
model_command::list(const command_context& context)
{
    auto root_path = context.root_path / default_path;

    for (auto const& filesystem_entry : std::filesystem::directory_iterator(root_path)) {
        if (!std::filesystem::is_directory(filesystem_entry)) {
            continue;
        }

        auto manifest_path = filesystem_entry.path() / manifest_name;
        auto manifest_document = tomlfile<manifest>::read(filesystem_entry / manifest_path);

        std::cout << filesystem_entry.path().filename().string() << '\t';
        std::cout << manifest_document.model.architecture << '\t';
        std::cout << manifest_document.model.partitioning << '\t';
        std::cout << manifest_document.model.repository << std::endl;
    }
}


void
model_command::remove(const command_context& context)
{
    auto repo_path = context.root_path / default_path / _M_id;
    std::filesystem::remove_all(repo_path);
}


} // namespace runtime
} // namespace metalchat

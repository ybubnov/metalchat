// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/huggingface.h>
#include <metalchat/repository.h>

#include "credential.h"
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
  _M_config("config"),
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
    push_handler(_M_pull, [&](const command_context& c) { pull(c); });

    _M_list.add_description("list the available models");
    _M_list.add_argument("--abbrev")
        .help(("instead of showing the full model identifier, "
               "show a prefix than names a model uniquely"))
        .flag();
    push_handler(_M_list, [&](const command_context& c) { list(c); });

    _M_remove.add_description("remove matching models");
    _M_remove.add_argument("id").help("a model identifier").required().store_into(_M_id);
    push_handler(_M_remove, [&](const command_context& c) { remove(c); });

    _M_config.add_description("get and set model run options");
    _M_config.add_argument("id").help("a model identifier").required().store_into(_M_id);
    _M_config.add_argument("name")
        .help("name of the target option")
        .metavar("<name>")
        .required()
        .store_into(_M_config_name)
        .nargs(1);
    _M_config.add_argument("value")
        .help("value of the target option")
        .metavar("<value>")
        .store_into(_M_config_value)
        .nargs(0, 1);
    push_handler(_M_config, [&](const command_context& c) { config(c); });
}


void
model_command::pull(const command_context& context)
{
    manifest manifest_document = {
        .model =
            {.repository = _M_repository,
             .variant = _M_variant,
             .architecture = _M_arch,
             .partitioning = _M_partitioning}
    };

    url repo_url(_M_repository);

    auto repo_path = context.root_path / default_path / manifest_document.model.id();
    auto manifest_path = repo_path / manifest_name;

    if (std::filesystem::exists(repo_path)) {
        throw std::invalid_argument("pull: model already exists");
    }

    std::cout << "Pulling from '" << _M_repository << "'..." << std::endl;

    using filesystem_type = http_tracking_filesystem;
    using transformer_type = huggingface::llama3;
    using http_repository = huggingface_repository<transformer_type, filesystem_type>;

    http_bearer_auth<keychain_provider> http_auth;
    http_tracking_filesystem filesystem(repo_url, http_auth);
    http_repository repository(repo_url.path(), repo_path, filesystem);

    repository.clone();

    tomlfile<manifest> manifest_file(manifest_path, tomlformat::multiline);
    manifest_file.write(manifest_document);
}


void
model_command::list(const command_context& context)
{
    auto root_path = context.root_path / default_path;
    bool use_abbrev = _M_list.get<bool>("--abbrev");

    for (auto const& filesystem_entry : std::filesystem::directory_iterator(root_path)) {
        if (!std::filesystem::is_directory(filesystem_entry)) {
            continue;
        }

        auto manifest_path = filesystem_entry.path() / manifest_name;
        auto manifest_document = tomlfile<manifest>::read(filesystem_entry / manifest_path);

        std::cout << ansi::yellow;
        if (use_abbrev) {
            std::cout << manifest_document.model.abbrev_id() << '\t';
        } else {
            std::cout << manifest_document.model.id() << '\t';
        }

        std::cout << ansi::reset;
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


void
model_command::config(const command_context& context)
{
    auto repo_path = context.root_path / default_path / _M_id;
    auto manifest_path = repo_path / manifest_name;

    tomlfile<manifest> manifest_file(manifest_path, tomlformat::multiline);
    auto manifest_document = manifest_file.read();

    if (_M_config_value.empty()) {
        auto config_value = manifest_document.model.config.and_then(
            [&](auto& config) -> std::optional<std::string> {
            if (auto it = config.find(_M_config_name); it != config.end()) {
                return it->second;
            }
            return std::nullopt;
        }
        );

        if (config_value) {
            std::cout << config_value.value() << std::endl;
        } else {
            // Throw an exception with an empty error string, so that the
            // program only returns a non-zero status code without printing
            // any error information.
            throw std::invalid_argument("");
        }
    } else {
        // Ensure that model supports this option.
        using optional_config_type = decltype(manifest_document.model.config);
        using config_type = optional_config_type::value_type;

        auto config = manifest_document.model.config.value_or(config_type());
        config.insert_or_assign(_M_config_name, _M_config_value);

        manifest_document.model.config = config;
        manifest_file.write(manifest_document);
    }
}


} // namespace runtime
} // namespace metalchat

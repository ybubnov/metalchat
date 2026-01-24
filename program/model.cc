// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/huggingface.h>
#include <metalchat/repository.h>

#include "credential.h"
#include "http.h"
#include "manifest.h"
#include "model.h"


namespace metalchat {
namespace runtime {


model_provider::model_provider(const std::filesystem::path& p)
: _M_path(p / default_path)
{}


std::filesystem::path
model_provider::resolve_path(const std::string& id) const
{
    return _M_path / id;
}


bool
model_provider::exists(const std::string& id) const
{
    auto model_path = resolve_path(id);
    return std::filesystem::exists(model_path);
}


model_info
model_provider::find(const std::string& id) const
{
    auto model_path = resolve_path(id);
    if (!std::filesystem::exists(model_path)) {
        throw std::invalid_argument(std::format("fatal: model '{}' not found", id));
    }

    auto m = ManifestFile::read(model_path / manifest::default_name);
    if (m.id() != id) {
        throw std::runtime_error(std::format("fatal: model '{}' is corrupted", id));
    }

    return model_info{.manifest = m, .path = model_path};
}


void
model_provider::remove(const std::string& id)
{
    // Ensure that model exists before deleting it from the repository.
    auto model = find(id);
    std::filesystem::remove_all(model.path);
}


void
model_provider::update(const model_info& m)
{
    auto model = find(m.manifest.id());
    auto manifest_path = model.path / manifest::default_name;

    ManifestFile file(manifest_path, tomlformat::multiline);
    file.write(m.manifest);
}


void
model_provider::insert(const manifest& m)
{
    auto model_id = m.id();
    auto model_path = resolve_path(model_id);
    auto manifest_path = model_path / manifest::default_name;

    if (exists(model_id)) {
        throw std::invalid_argument("fatal: model already exists");
    }

    std::cout << "Pulling from '" << m.model.repository << "'..." << std::endl;

    // TODO: add support of different model types.
    using filesystem_type = http_tracking_filesystem;
    using transformer_type = huggingface::llama3;
    using http_repository = huggingface_repository<transformer_type, filesystem_type>;

    url repo_url(m.model.repository);

    http_bearer_auth<keychain_provider> http_auth;
    http_tracking_filesystem filesystem(repo_url, http_auth);
    http_repository repository(repo_url.path(), model_path, filesystem);

    repository.clone();

    ManifestFile file(manifest_path, tomlformat::multiline);
    file.write(m);
}


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
}


void
model_command::pull(const command_context& context)
{
    manifest m = {
        .model =
            {.repository = _M_repository,
             .variant = _M_variant,
             .architecture = _M_arch,
             .partitioning = _M_partitioning}
    };

    model_provider models(context.root_path);
    models.insert(m);
}


void
model_command::list(const command_context& context)
{
    auto root_path = context.root_path / model_provider::default_path;
    bool use_abbrev = _M_list.get<bool>("--abbrev");

    model_provider models(context.root_path);
    models.find_if([&](const model_info& m) -> bool {
        std::cout << ansi::yellow;
        if (use_abbrev) {
            std::cout << m.manifest.abbrev_id() << "  ";
        } else {
            std::cout << m.manifest.id() << "  ";
        }

        std::cout << ansi::reset;
        std::cout << m.manifest.model.architecture << "  ";
        std::cout << m.manifest.model.partitioning << "  ";
        std::cout << m.manifest.model.repository << std::endl;

        return false;
    });
}


void
model_command::remove(const command_context& context)
{
    model_provider models(context.root_path);
    models.remove(_M_id);
}


} // namespace runtime
} // namespace metalchat

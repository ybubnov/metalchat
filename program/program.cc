// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cstdlib>

#include <metalchat/metalchat.h>


#include "config.h"
#include "model.h"
#include "program.h"


namespace metalchat {
namespace runtime {


program::program()
: basic_command("metalchat", __lib_metalchat_version),
  _M_credential(*this),
  _M_model(*this),
  _M_options(*this),
  _M_stdin("-"),
  _M_checkout("checkout"),
  _M_model_id()
{
    auto config_path = std::filesystem::path("~") / default_path / default_config_path;

    _M_command.add_description("A self-sufficient runtime for large language models");
    _M_command.add_argument("-f", "--file")
        .help("read configuration file only from this location")
        .metavar("<config-file>")
        .default_value(config_path.string())
        .nargs(1);

    _M_stdin.add_description("read from stdin and run model inference");
    _M_stdin.add_argument("model").help("the model to launch for the input processing").nargs(0, 1);
    _M_stdin.add_argument("--local").help("use a current working directory manifest").flag();
    _M_stdin.add_argument("--global").help("use a global manifest").flag();
    push_handler(_M_stdin, [&](const command_context& c) { handle_stdin(c); });

    _M_checkout.add_description("switch models");
    _M_checkout.add_argument("--local").help("use a current working directory manifest").flag();
    _M_checkout.add_argument("--global").help("use a global manifest").flag();
    _M_checkout.add_argument("model")
        .help("the model to prepare for working")
        .required()
        .store_into(_M_model_id);
    push_handler(_M_checkout, [&](const command_context& c) { handle_checkout(c); });
}


void
program::handle_stdin(const command_context& context)
{
    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    auto manifest_file = context.resolve_manifest(resolve_scope(_M_stdin));
    auto scope_path = manifest_file.path().parent_path();
    auto scope_manifest = manifest{};

    auto model_id = _M_stdin.present("model");
    if (!model_id) {
        scope_manifest = manifest_file.read();
        model_id = scope_manifest.id();
    }

    model_provider models(context.root_path);
    auto model = models.find(model_id.value());

    auto repository = filesystem_repository<Transformer>(model.path);
    auto tokenizer = repository.retrieve_tokenizer();
    auto options = repository.retrieve_options();

    if (scope_manifest.options) {
        auto scope_options = scope_manifest.options.value();
        auto first = scope_options.begin();
        auto last = scope_options.end();

        options = TransformerTraits::merge_options(first, last, options);
    }

    auto transformer = repository.retrieve_transformer(options);
    auto interp = metalchat::interpreter(transformer, tokenizer);

    const std::size_t max_input_size = 1024;
    std::string input(max_input_size, '\0');

    if (std::cin.read(input.data(), max_input_size)) {
        throw std::runtime_error("failed reading from stdin");
    }
    input = std::string(input.c_str(), std::cin.gcount());

    if (auto system_prompt = scope_manifest.system_prompt(scope_path); system_prompt) {
        interp.write(basic_message("system", system_prompt.value()));
    }
    interp.write(basic_message("user", input));

    // TODO: ensure that encoded context does not exceed the model limit.
    std::ostream_iterator<std::string> content_iterator(std::cout << std::unitbuf);
    interp.read(content_iterator);
}


void
program::handle_checkout(const command_context& context)
{
    model_provider models(context.root_path);
    auto model = models.find(_M_model_id);

    auto scope = resolve_scope(_M_checkout);
    auto manifest_file = context.resolve_manifest(scope, /*missing_ok=*/true);

    manifest m = {};
    if (manifest_file.exists()) {
        m = manifest_file.read();
    }

    m.model = model.manifest.model;
    manifest_file.write(m);
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
    auto global_path = root_path / manifest::workspace_name;
    auto local_path = std::filesystem::current_path() / manifest::workspace_name;

    std::filesystem::create_directories(root_path);
    using manifest_file = command_context::manifest_file;

    std::unordered_map<command_scope, manifest_file> manifests = {
        {context_scope::local, manifest_file(local_path, tomlformat::multiline)},
        {context_scope::global, manifest_file(global_path, tomlformat::multiline)},
    };

    command_context context{
        .root_path = root_path,
        .config_file = tomlfile<config>(config_path),
        .manifests = manifests
    };
    basic_command::handle(context);
}


} // namespace runtime
} // namespace metalchat

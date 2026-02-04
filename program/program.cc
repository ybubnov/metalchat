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
  _M_prompt("prompt"),
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

    add_scope_arguments(_M_stdin);
    _M_stdin.add_description("read from stdin and run model inference");
    _M_stdin.add_argument("model").help("the model to launch for the input processing").nargs(0, 1);
    push_handler(_M_stdin, [&](const command_context& c) { handle_stdin(c); });

    // It is possible to put both, `-c` and `promptfile` into a mutually exclusive
    // group, but argparse library renders really ugly help for such a setup, so
    // just try to parse `-c` first, then `promptfile, throw otherwise.
    add_scope_arguments(_M_prompt);
    _M_prompt.add_description("read prompt and run model inference");
    _M_prompt.add_argument("-c")
        .help("pass user prompt as a string")
        .metavar("<command>")
        .nargs(0, 1);
    _M_prompt.add_argument("promptfile")
        .help("read prompt from file and pass it to the model")
        .nargs(0, 1);
    push_handler(_M_prompt, [&](const command_context& c) { handle_prompt(c); });

    add_scope_arguments(_M_checkout);
    _M_checkout.add_description("switch between different models");
    _M_checkout.add_argument("model")
        .help("the model to prepare for working")
        .required()
        .store_into(_M_model_id);
    push_handler(_M_checkout, [&](const command_context& c) { handle_checkout(c); });
}


program_scope
program::resolve_program_scope(const command_context& context, const parser_type& parser) const
{
    auto manifest_file = resolve_manifest(context, parser);
    auto manifest_path = manifest_file.path().parent_path();
    auto manifest = manifest_file.read();

    model_provider models(context.root_path);
    auto model = models.find(manifest.id());

    return program_scope{manifest_path, model.path, manifest};
}


program_scope
program::resolve_program_scope(const command_context& context, const std::string& model_id) const
{
    model_provider models(context.root_path);
    auto model = models.find(model_id);
    return program_scope{model.path, model.path, model.manifest};
}


void
program::transform(const program_scope& scope, const std::string& prompt) const
{
    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    scoped_repository_adapter<Transformer> repo(scope.repo_path, scope.manifest);
    auto transformer = repo.retrieve_transformer();
    auto tokenizer = repo.retrieve_tokenizer();

    auto interp = metalchat::interpreter(transformer, tokenizer);
    auto system_prompt = scope.manifest.system_prompt(scope.path);
    if (system_prompt) {
        interp.write(basic_message("system", system_prompt.value()));
    }
    interp.write(basic_message("user", prompt));

    // TODO: ensure that encoded context does not exceed the model limit.
    std::ostream_iterator<std::string> content_iterator(std::cout << std::unitbuf);
    interp.read(content_iterator);
}


void
program::handle_prompt(const command_context& context)
{
    std::string input;

    if (auto prompt = _M_prompt.present("-c"); prompt) {
        input = prompt.value();
    } else if (auto filename = _M_prompt.present("promptfile"); filename) {
        std::ifstream prompt_file(filename.value());
        if (!prompt_file.is_open()) {
            auto error = std::format("error: failed reading from '{}' file", filename.value());
            throw std::runtime_error(error);
        }

        using iterator = std::istreambuf_iterator<char>;
        input = std::string(iterator(prompt_file), iterator());
    } else {
        throw std::runtime_error("error: either command prompt or prompt file is required");
    }

    auto scope = resolve_program_scope(context, _M_prompt);
    transform(scope, input);
}


void
program::handle_stdin(const command_context& context)
{
    const std::size_t max_input_size = 1024;
    std::string input(max_input_size, '\0');

    if (std::cin.read(input.data(), max_input_size)) {
        throw std::runtime_error("error: failed reading from stdin");
    }
    input = std::string(input.c_str(), std::cin.gcount());

    program_scope scope;
    if (auto model_id = _M_stdin.present("model"); model_id) {
        scope = resolve_program_scope(context, model_id.value());
    } else {
        scope = resolve_program_scope(context, _M_stdin);
    }

    transform(scope, input);
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
    if (!_M_command) {
        std::cout << _M_command;
        throw std::runtime_error("");
    }

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

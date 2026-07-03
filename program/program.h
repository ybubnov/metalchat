// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cstdio>
#include <string_view>

#include <metalchat/huggingface/llama.h>
#include <metalchat/metalchat.h>

#include "credential.h"
#include "model.h"
#include "options.h"


namespace metalchat {
namespace runtime {


struct program_scope {
    std::filesystem::path path;
    std::filesystem::path repo_path;
    manifest manifest;
};


/// This is the main entrypoint of the metalchat command line program.
///
/// On creation, this method registers all of the necessary sub-commands and their handlers.
class program : public basic_command {
public:
    static constexpr std::string_view default_path = ".metalchat";
    static constexpr std::string_view default_config_path = "config.toml";

    program();

    void
    handle(int argc, char** argv);

    void
    handle_stdin(const command_context&);

    void
    handle_checkout(const command_context&);

    void
    handle_prompt(const command_context&);

private:
    /// Loads an existing model (based on the configured scope) and runs it
    /// by prompting data specified in the stream.
    template <typename Transformer>
    static void
    transform(const program_scope& scope, const std::string& prompt)
    {
        using Formatter = Transformer::formatter_type;
        using Message = Formatter::message_type;

        scoped_repository_adapter<Transformer> repo(scope.repo_path, scope.manifest);
        auto transformer = repo.retrieve_transformer();
        auto tokenizer = repo.retrieve_tokenizer();

        using Tokenizer = decltype(tokenizer);
        using TokenizerTraits = text::tokenizer_traits<Tokenizer>;

        auto formatter = Formatter(tokenizer);
        auto interp = metalchat::interpreter(transformer, formatter);

        auto system_prompt = scope.manifest.system_prompt(scope.path);
        if (system_prompt) {
            interp.write(Message(role::system, system_prompt.value()));
        }
        interp.write(Message(role::request, prompt));

        // TODO: ensure that encoded context does not exceed the model limit.
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        interp.read(std::cout);
    }

    template <std::size_t... Indices>
    void
    register_supported_architechtures(std::index_sequence<Indices...>)
    {
        (register_supported_architechture<Indices>(), ...);
    }

    template <std::size_t Index>
    void
    register_supported_architechture()
    {
        using Arch = architecture_typeinfo<Index>;
        using Transformer = Arch::transformer_type;

        _M_transformers.insert_or_assign(std::string(Arch::name), &transform<Transformer>);
    }

    program_scope
    resolve_program_scope(const command_context& context, const parser_type& parser) const;

    program_scope
    resolve_program_scope(const command_context& context, const std::string& model_id) const;

    std::string _M_model_id;

    parser_type _M_stdin;
    parser_type _M_prompt;
    parser_type _M_checkout;

    credential_command _M_credential;
    model_command _M_model;
    options_command _M_options;

    using Transformer = std::function<decltype(transform<huggingface::llama3>)>;
    std::unordered_map<std::string, Transformer> _M_transformers;
};


} // namespace runtime
} // namespace metalchat

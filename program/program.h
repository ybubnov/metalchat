// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string_view>

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
    void
    transform(const program_scope& scope, const std::string& prompt) const;

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
};


} // namespace runtime
} // namespace metalchat

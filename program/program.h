// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string_view>

#include "chat.h"
#include "credential.h"
#include "model.h"


namespace metalchat {
namespace runtime {


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

private:
    parser_type _M_stdin;

    credential_command _M_credential;
    model_command _M_model;
};


} // namespace runtime
} // namespace metalchat

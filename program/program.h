// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <string_view>

#include "chat.h"
#include "credential.h"
#include "model.h"


namespace metalchat {
namespace program {


/// This is the main entrypoint of the metalchat command line program.
///
/// On creation, this method registers all of the necessary sub-commands and their handlers.
class program : public basic_command {
public:
    static constexpr std::string_view default_config_path = ".metalchat/config";

    program();

    void
    handle(int argc, char** argv);

private:
    credential_command _M_credential;
    model_command _M_model;
};


} // namespace program
} // namespace metalchat

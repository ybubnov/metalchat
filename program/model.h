// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <toml.hpp>

#include "command.h"


namespace metalchat {
namespace program {


struct architecture {
    static std::string llama3x2_1b;
    static std::string llama3x2_3b;
};


struct manifest {
    std::string architecture;
    bool sharded;
};


} // namespace program
} // namespace metalchat


TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::program::manifest, architecture, sharded);


namespace metalchat {
namespace program {

class model_command : public basic_command {
public:
    model_command(basic_command& parent);

    void
    pull(const command_context&);

    void
    list(const command_context&);

    void
    remove(const command_context&);

private:
    parser_type _M_pull;
    parser_type _M_list;
    parser_type _M_remove;

    std::string _M_repository;
    std::string _M_arch;
    std::string _M_name;
};


} // namespace program
} // namespace metalchat

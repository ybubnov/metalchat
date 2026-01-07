// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include "command.h"


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

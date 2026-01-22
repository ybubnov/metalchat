// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include "command.h"
#include "model.h"


namespace metalchat {
namespace runtime {


class options_command : public basic_command {
public:
    options_command(basic_command& parent);

    void
    get(const command_context&);

    void
    set(const command_context&);

    void
    unset(const command_context&);

private:
    parser_type _M_get;
    parser_type _M_set;
    parser_type _M_unset;

    std::string _M_name;
    std::string _M_value;
    std::string _M_id;
};


} // namespace runtime
} // namespace metalchat

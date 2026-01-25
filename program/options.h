// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include "command.h"
#include "model.h"


namespace metalchat {
namespace runtime {


struct optionkind {
    static std::string integer;
    static std::string boolean;
    static std::string floating;
    static std::string string;
};


class options_command : public basic_command {
public:
    options_command(basic_command& parent);

    void
    get(const command_context&) const;

    void
    set(const command_context&) const;

    void
    unset(const command_context&) const;

private:
    tomlfile<manifest>
    resolve_manifest(const command_context&) const;

    parser_type _M_get;
    parser_type _M_set;
    parser_type _M_unset;

    std::string _M_name;
    std::string _M_value;
    std::string _M_type;
};


} // namespace runtime
} // namespace metalchat

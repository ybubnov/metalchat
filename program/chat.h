// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <functional>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include <metalchat/metalchat.h>

#include "command.h"


namespace metalchat {
namespace workspace {


struct chat_create_options {
    std::optional<std::string> name = std::nullopt;
    std::optional<std::string> system_prompt = std::nullopt;
    std::string model = "";
    std::string arch = "";
    std::string impl = "";
};


class chat {
public:
    chat(const chat_create_options&);
};


class chat_command : public basic_command {
public:
    chat_command(basic_command& command);

    void
    create();

private:
    chat_create_options _M_create_options;
};


} // namespace workspace
} // namespace metalchat

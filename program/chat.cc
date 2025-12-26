// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "chat.h"


namespace metalchat {
namespace program {


chat_command::chat_command(CLI::App& app)
{
    auto chat = app.add_subcommand("chat", "Manage and chat with language models");
    auto chat_create = chat->add_subcommand("create", "Create a new chat");
    auto chat_list = chat->add_subcommand("list", "List available chats");
    auto chat_continue = chat->add_subcommand("continue", "Continue started chat");
    auto chat_remove = chat->add_subcommand("remove", "Remove chats");
}


} // namespace program
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cerrno>
#include <cstdlib>
#include <filesystem>

#include <metalchat/metalchat.h>
#include <replxx.hxx>

#include "chat.h"


namespace metalchat {
namespace internal {


chat::chat(const chat_create_options&) {}


chat_command::chat_command(basic_command& parent)
: basic_command("chat"),
  _M_create_options()
{
    // auto chat = app.add_subcommand("chat", "Manage and chat with language models");
    // auto chat_create = chat->add_subcommand("create", "Create a new chat");
    // chat_create->add_option("--name", _M_create_options.name, "Assign a name to a chat");
    // chat_create->add_option("--arch", _M_create_options.arch, "A model architecture")
    //     ->default_val(architecture::llama3x2_1b);
    // chat_create->add_option("--impl", _M_create_options.impl, "A model implementation")
    //     ->default_val(implementation::reference);
    // chat_create->add_option("model", _M_create_options.model, "A model model for a chat")
    //     ->default_val(chat::default_model);
    // chat_create->callback([&]() { create(); });

    // auto chat_list = chat->add_subcommand("list", "List available chats");
    // auto chat_continue = chat->add_subcommand("continue", "Continue started chat");
    // auto chat_remove = chat->add_subcommand("remove", "Remove chats");
}


void
chat_command::create()
{
    replxx::Replxx shell;

    auto repo_path = std::filesystem::path(_M_create_options.model);
    auto repository = filesystem_repository<huggingface::llama3>(repo_path);

    auto tokenizer = repository.retrieve_tokenizer();
    auto transformer = repository.retrieve_transformer();

    auto interp = metalchat::interpreter(transformer, tokenizer);

    for (;;) {
        char const* raw_input = nullptr;
        do {
            raw_input = shell.input("(metalchat): ");
        } while ((raw_input == nullptr) && (errno == EAGAIN));

        if (raw_input == nullptr) {
            break;
        }

        std::string input(raw_input);
        interp.write(metalchat::basic_message("user", input));

        std::cout << interp.read_text() << std::endl;
    }
}


} // namespace internal
} // namespace metalchat

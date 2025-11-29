// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cerrno>
#include <cstdlib>
#include <filesystem>

#include <metalchat/metalchat.h>
#include <replxx.hxx>


int
main()
{
    replxx::Replxx shell;

    auto weights_path = std::filesystem::path(std::getenv("METALCHAT_SAFETENSOR_PATH"));
    auto tokens_path = std::filesystem::path(std::getenv("METALCHAT_TOKENIZER_PATH"));

    auto options = metalchat::nn::default_llama3_1b_options().heap_size(0);
    auto interp = metalchat::make_llama3(weights_path, tokens_path, options);

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

    return 0;
}

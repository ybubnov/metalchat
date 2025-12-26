// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <memory>

#include <CLI/CLI.hpp>
#include <metalchat/metalchat.h>
#include <replxx.hxx>

#include "chat.h"
#include "model.h"
#include "remote.h"


static const std::string config_path = "~/.metalchat/config";


int
main(int argc, char** argv)
{
    std::string config_option;

    CLI::App app("A self-sufficient runtime for large language models", "metalchat");
    app.add_option("--config", config_option)->default_val(config_path);

    metalchat::program::remote_command remote(app);
    metalchat::program::model_command model(app);
    metalchat::program::chat_command chat(app);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    /*
    replxx::Replxx shell;

    huggingface_repository repo("meta-llama/Llama3.2-1B-Instruct");

    auto repo_path = std::filesystem::path(std::getenv("METALCHAT_PATH"));
    auto tokens_path = repo_path / "original/tokenizer.model";

    metalchat::reference::llama3_autoloader loader(repo_path);
    metalchat::text::bpe tokenizer(tokens_path);

    auto options = metalchat::nn::default_llama3_1b_options();
    auto transformer = loader.load(options);
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
    */

    return 0;
}

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <CLI/CLI.hpp>

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

    return 0;
}

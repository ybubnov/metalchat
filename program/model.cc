// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "model.h"


namespace metalchat {
namespace program {


model_command::model_command(CLI::App& app)
{
    auto model = app.add_subcommand("model", "Manage language models");
    auto model_pull = model->add_subcommand("pull", "Download a model from a remote server");
    model_pull->callback([&]() { pull(); });

    auto model_list = model->add_subcommand("list", "List the available models");
    model_list->callback([&]() { list(); });

    auto model_remove = model->add_subcommand("remove", "Remove models");
    model_remove->callback([&]() { remove(); });
}


void
model_command::pull()
{}


void
model_command::list()
{}


void
model_command::remove()
{}


git_model::git_model(const std::string& repo) {}


} // namespace program
} // namespace metalchat

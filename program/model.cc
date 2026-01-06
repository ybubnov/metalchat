// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "model.h"


namespace metalchat {
namespace program {


model_command::model_command(basic_command& parent)
: basic_command("model", parent),
  _M_pull("pull"),
  _M_list("list"),
  _M_remove("remove")
{
    _M_command.add_description("manage language models");

    _M_pull.add_description("download a model from a remote server");
    _M_list.add_description("list the available models");
    _M_remove.add_description("remove matching models");

    push_handler(_M_pull, [&] { pull(); });
    push_handler(_M_list, [&] { list(); });
    push_handler(_M_remove, [&] { remove(); });
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


} // namespace program
} // namespace metalchat

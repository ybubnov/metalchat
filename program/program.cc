// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "program.h"


namespace metalchat {
namespace program {


program::program()
: basic_command("metalchat"),
  _M_credential(*this),
  _M_model(*this)
{
    _M_command.add_description("A self-sufficient runtime for large language models");
}


void
program::handle(int argc, char** argv)
{
    _M_command.parse_args(argc, argv);
    basic_command::handle();
}


} // namespace program
} // namespace metalchat

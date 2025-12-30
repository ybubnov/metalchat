// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/text/bpe.h>


namespace metalchat {
namespace text {


std::string
make_reserved_token(int32_t index)
{
    return std::format("<|reserved_special_token_{}|>", index);
}


} // namespace text
} // namespace metalchat

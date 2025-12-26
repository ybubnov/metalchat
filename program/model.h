// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string>


namespace metalchat {
namespace program {


class local_model {};


class git_model {
public:
    git_model(const std::string& repo);
};


} // namespace program
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <optional>
#include <string>

#include <CLI/CLI.hpp>


namespace metalchat {
namespace program {


class remote {
public:
    remote(const std::string& host);

    void
    add(const std::string& token);

private:
    std::string _M_host;
};


class remote_command {
public:
    struct add_options {
        std::string hostname = "";
        std::optional<std::string> token = std::nullopt;
    };

    struct remove_options {
        std::string hostname = "";
    };

    remote_command(CLI::App& app);

    void
    add();

    void
    list();

    void
    remove();

private:
    add_options _M_add_options;
    remove_options _M_remove_options;
};


} // namespace program
} // namespace metalchat

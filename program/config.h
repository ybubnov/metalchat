// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <map>
#include <string>


namespace metalchat {
namespace program {


/// The credential store in a configuration file.
struct credential_configuration {
    std::string username;
    std::string backend;
};


struct configuration {
    /// Credentials contain a mapping of protocol+hostname to the user
    std::map<std::string, credential_configuration> credenitals;
};


class configuration_repository {
public:
    configuration_repository(const std::filesystem::path& p);

    void
    store(const configuration& c) const;

    configuration
    load() const;

private:
    std::filesystem::path _M_path;
};


} // namespace program
} // namespace metalchat

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <iostream>


namespace metalchat {
namespace runtime {


struct hard_linking_filesystem {
    void
    read(const std::string& filename, std::ostream& output) const
    {
        std::ofstream filestream(filename);
        output << filestream.rdbuf();
    }

    void
    copy(const std::string& src, const std::string& dst) const
    {
        std::filesystem::create_hard_link(src, dst);
    }

    bool
    exists(const std::string& filename) const
    {
        return std::filesystem::exists(filename);
    }
};


} // namespace runtime
} // namespace metalchat

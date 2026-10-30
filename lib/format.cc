// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <filesystem>
#include <format>
#include <source_location>

#include <metalchat/tensor/format.h>


namespace metalchat {


std::string
format_source_location(const std::source_location& source_location)
{
    const auto source_filepath = std::filesystem::path(source_location.file_name());
    const auto source_file = source_filepath.filename().string();
    const auto source_line = source_location.line();

    return std::vformat("{}#{}:", std::make_format_args(source_file, source_line));
}


} // namespace metalchat

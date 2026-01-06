// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "config.h"


namespace metalchat {
namespace program {


configuration_repository::configuration_repository(const std::filesystem::path& p)
: _M_path(p)
{}


void
configuration_repository::store(const configuration& c) const
{}


configuration
configuration_repository::load() const
{
    return configuration{};
}


} // namespace program
} // namespace metalchat

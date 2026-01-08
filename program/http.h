// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <curl/curl.h>


namespace metalchat {
namespace program {


class url {
public:
    url(const std::string& u);
    ~url();

    std::string
    protocol() const;

    std::string
    host() const;

private:
    std::string
    part(CURLUPart p) const;

    CURLU* _M_url = nullptr;
};


} // namespace program
} // namespace metalchat

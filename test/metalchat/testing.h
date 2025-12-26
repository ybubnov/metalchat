// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>


#ifndef __lib_metalchat_test_fixture_directory
#define __lib_metalchat_test_fixture_directory "test_fixture"
#endif


std::filesystem::path
test_fixture_path()
{
    return std::filesystem::path(__lib_metalchat_test_fixture_directory);
}

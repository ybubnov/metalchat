// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>


#ifndef METALCHAT_TEST_FIXTURE_DIRECTORY
#define METALCHAT_TEST_FIXTURE_DIRECTORY "test_fixture"
#endif


std::filesystem::path
test_fixture_path()
{
    return std::filesystem::path(METALCHAT_TEST_FIXTURE_DIRECTORY);
}

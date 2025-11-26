// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>


#ifndef METALCHAT_TESTDATA_DIRECTORY
#define METALCHAT_TESTDATA_DIRECTORY "testdata"
#endif


std::filesystem::path
testdata_path()
{
    return std::filesystem::path(METALCHAT_TESTDATA_DIRECTORY);
}

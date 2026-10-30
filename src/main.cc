// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "program.h"


int
main(int argc, char** argv)
{
    metalchat::runtime::program program;

    try {
        program.handle(argc, argv);
    } catch (const std::exception& e) {
        std::string what(e.what());
        if (!what.empty()) {
            std::cout << what << std::endl;
        }
        std::exit(1);
    }

    return 0;
}

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "program.h"


namespace mcp = metalchat::program;


int
main(int argc, char** argv)
{
    mcp::program program;

    try {
        program.handle(argc, argv);
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        std::exit(1);
    }

    return 0;
}

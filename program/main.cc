// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cerrno>

#include <metalchat/metalchat.h>
#include <replxx.hxx>


int
main()
{
    replxx::Replxx shell;

    for (;;) {
        char const* raw_input = nullptr;
        do {
            raw_input = shell.input("(metalchat): ");
        } while ((raw_input == nullptr) && (errno == EAGAIN));

        if (raw_input == nullptr) {
            break;
        }
        std::string input(raw_input);
        std::cout << "'" << input << "'" << std::endl;
    }

    return 0;
}

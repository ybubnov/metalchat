// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/metalchat.h>

using namespace metalchat;

int
main()
{
    hardware_accelerator gpu0;
    nn::indirect_layer<nn::linear<float>> linear(10, 64, gpu0);

    auto input = metalchat::rand<float>({14, 10}, gpu0);
    auto output = linear(input);

    std::cout << output.get() << std::endl;

    return 0;
}

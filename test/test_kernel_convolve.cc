// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/convolve.h>
#include <metalchat/tensor.h>

using namespace metalchat;


TEST_CASE("Test conv1d fixed input", "[kernel::conv1d]")
{
    hardware_accelerator gpu0;
    kernel::conv1d<float> conv(gpu0);

    std::size_t in_channels = 16;
    std::size_t out_channels = 32;
    std::size_t groups = 16;
    std::size_t input_size = 21;

    auto input = shared_tensor(full<float>({in_channels, input_size}, 2.0f));
    auto weight = shared_tensor(full<float>({out_channels, in_channels / groups, 5}, 0.5f));

    auto output = conv(input, weight, /*padding=*/0, /*groups=*/groups).get();

    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == out_channels);
    REQUIRE(output.size(1) == 17);

    std::cout << output << std::endl;
    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            REQUIRE_THAT((output[i, j]), Catch::Matchers::WithinAbs(5.0, 0.0001));
        }
    }
}


TEST_CASE("Test conv1d random input", "[kernel::conv1d]")
{
    hardware_accelerator gpu0;
    kernel::conv1d<float> conv(gpu0);

    auto input = shared_tensor(rand<float>({10, 1024, 66}));
    auto weight = shared_tensor(rand<float>({1024, 1, 3}));

    auto output = conv(input, weight, /*padding=*/1, /*groups=*/1024);

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 10);
    REQUIRE(output.size(1) == 1024);
    REQUIRE(output.size(2) == 66);
}

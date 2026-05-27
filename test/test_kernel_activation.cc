// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <cmath>
#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/activation.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("SiLU function", "[kernel::silu]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::silu<float> silu(gpu0);

    auto input = shared_tensor(rand<float>({3, 5, 8192}));

    auto output = silu(input).get();
    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 5);
    REQUIRE(output.size(2) == 8192);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                auto result = input[i, j, k] / (1 + std::exp(-input[i, j, k]));
                REQUIRE_THAT((output[i, j, k]), Catch::Matchers::WithinAbs(result, 0.00001));
            }
        }
    }
}


TEST_CASE("GELU function", "[kernel::gelu]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::gelu<float> gelu(gpu0);

    auto input = shared_tensor(rand<float>({3, 5, 8192}));

    auto output = gelu(input).get();
    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 5);
    REQUIRE(output.size(2) == 8192);

    const auto sqrt_2_pi = std::sqrt(0.5 * std::numbers::pi);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                const auto x = input[i, j, k];
                const auto xh = x + 0.044715 * x * x * x;
                const auto result = x * 0.5 * (1.0 + std::tanh(sqrt_2_pi * xh));

                REQUIRE_THAT((output[i, j, k]), Catch::Matchers::WithinAbs(result, 0.00001));
            }
        }
    }
}

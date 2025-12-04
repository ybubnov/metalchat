// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Hadamard product", "[kernel::hadamard]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::hadamard<float> m(gpu0);

    auto input1 = shared_tensor(rand<float>({3, 5, 8192}));
    auto input2 = shared_tensor(rand<float>({3, 5, 8192}));

    auto output = m(input1, input2).get();
    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 5);
    REQUIRE(output.size(2) == 8192);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                auto result = input1[i, j, k] * input2[i, j, k];
                REQUIRE_THAT((output[i, j, k]), Catch::Matchers::WithinAbs(result, 0.00001));
            }
        }
    }
}


TEST_CASE("Hadamard broadcasting product", "[kernel::hadamard_broadcast]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::hadamard_broadcast<float, int8_t, float> m(gpu0);

    auto weight = shared_tensor(rand<int8_t>({512, 64, 32}, 1, 10));
    auto scales = shared_tensor(rand<float>({512, 64, 1}));

    auto output = m(weight, scales).get();
    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 512);
    REQUIRE(output.size(1) == 64);
    REQUIRE(output.size(2) == 32);

    // std::cout << output.get() << std::endl;
    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                auto result = static_cast<float>(weight[i, j, k]);
                result *= static_cast<float>(scales[i, j, 0]);

                REQUIRE_THAT((output[i, j, k]), Catch::Matchers::WithinAbs(result, 0.00001));
            }
        }
    }
}


TEST_CASE("Scalar multiplication", "[kernel::scalar_mul]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::scalar_mul<float> m(gpu0);

    auto input = shared_tensor(rand<float>({1, 32, 4, 64}));
    auto output = m(input, 8.0f).get();

    REQUIRE(output.dim() == 4);
    for (std::size_t i = 0; i < output.dim(); i++) {
        REQUIRE(output.size(i) == input.size(i));
    }

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t m = 0; m < output.size(2); m++) {
                for (std::size_t n = 0; n < output.size(2); n++) {
                    REQUIRE_THAT(
                        (output[i, j, m, n]),
                        Catch::Matchers::WithinAbs(input[i, j, m, n] * 8.0f, 0.00001)
                    );
                }
            }
        }
    }
}

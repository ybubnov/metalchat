// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Test repeat interleave", "[functional::repeat_interleave]")
{
    auto original = shared_tensor(rand<float>({1, 6, 8, 64}));

    metalchat::hardware_accelerator gpu0;
    auto output = repeat_interleave(original, 4, /*dim=*/2, gpu0).get();

    REQUIRE(output.dim() == 5);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 6);
    REQUIRE(output.size(2) == 8);
    REQUIRE(output.size(3) == 4);
    REQUIRE(output.size(4) == 64);

    for (std::size_t i = 0; i < original.size(0); i++) {
        for (std::size_t j = 0; j < original.size(1); j++) {
            for (std::size_t k = 0; k < original.size(2); k++) {
                for (std::size_t l = 0; l < original.size(3); l++) {
                    for (std::size_t m = 0; m < output.size(3); m++) {
                        REQUIRE(original[i, j, k, l] == output[i, j, k, m, l]);
                    }
                }
            }
        }
    }
}

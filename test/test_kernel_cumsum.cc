// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/cumsum.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Cumulative sum", "[kernel::cumsum]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::cumsum<float, 16> cumsum(gpu0);

    auto input = shared_tensor(rand<float>({1, 1, 400}));
    auto output = cumsum(input).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == input.size(0));
    REQUIRE(output.size(1) == input.size(1));
    REQUIRE(output.size(2) == input.size(2));

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            std::vector<float> expect({input[i][j][0]});
            for (std::size_t k = 1; k < output.size(2); k++) {
                expect.push_back(input[i][j][k] + expect[k - 1]);
            }

            std::vector<float> actual(output[i][j].begin(), output[i][j].end());
            REQUIRE_THAT(actual, Catch::Matchers::Approx(expect).margin(0.0001));
        }
    }
}

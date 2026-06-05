// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor triu (diagonal=1)", "[functional::triu]")
{
    auto t = full<float>({10, 10}, 3.0f);
    triu(t, /*diagonal=*/1);

    for (std::size_t i = 0; i < t.size(0); i++) {
        for (std::size_t j = 0; j < t.size(1); j++) {
            if (j > i) {
                REQUIRE(t[i][j] == 3.0f);
            } else {
                REQUIRE(t[i][j] == 0.0f);
            }
        }
    }
}


TEST_CASE("Tensor triu (diagonal=4)", "[functional::triu]")
{
    auto t = full<float>({5, 10}, 3.0f);
    triu(t, /*diagonal=*/4);

    std::size_t last = 4;
    for (std::size_t i = 0; i < t.size(0); i++, last++) {
        for (std::size_t j = 0; j < t.size(1); j++) {
            if (j >= last) {
                REQUIRE(t[i][j] == 3.0f);
            } else {
                REQUIRE(t[i][j] == 0.0f);
            }
        }
    }
}


TEST_CASE("Tensor triu (diagonal=-4)", "[functional::triu]")
{
    auto t = full<float>({10, 10}, 3.0f);
    triu(t, /*diagonal=*/-4);

    std::size_t last = 1;
    for (std::size_t i = 0; i < t.size(0); i++) {
        for (std::size_t j = 0; j < t.size(1); j++) {
            if (i > 4 && j < last) {
                REQUIRE(t[i][j] == 0.0f);
            } else {
                REQUIRE(t[i][j] == 3.0f);
            }
        }
        if (i > 4) {
            last++;
        }
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <ranges>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor concatenate", "[concatenate]")
{
    auto t0 = shared_tensor(full<float>({3, 4, 2}, 1.0));
    auto t1 = shared_tensor(full<float>({3, 4, 2}, 2.0));
    auto t2 = shared_tensor(full<float>({3, 4, 2}, 3.0));
    auto t3 = shared_tensor(full<float>({3, 4, 2}, 4.0));
    auto t4 = shared_tensor(full<float>({3, 4, 2}, 5.0));

    metalchat::hardware_accelerator gpu0;

    auto tensors = {t0, t1, t2, t3, t4};

    auto output0 = concatenate(tensors, 0, gpu0).get();
    REQUIRE(output0.dim() == 3);
    REQUIRE(output0.size(0) == 15);
    REQUIRE(output0.size(1) == 4);
    REQUIRE(output0.size(2) == 2);
    REQUIRE(output0[0, 0, 0] == 1.0);
    REQUIRE(output0[14, 3, 1] == 5.0);

    auto output2 = concatenate(tensors, 2, gpu0).get();
    REQUIRE(output2.dim() == 3);
    REQUIRE(output2.size(0) == 3);
    REQUIRE(output2.size(1) == 4);
    REQUIRE(output2.size(2) == 10);
    REQUIRE(output2[0, 0, 0] == 1.0);
    REQUIRE(output2[2, 3, 9] == 5.0);
}


TEST_CASE("Tensor chunk", "[chunk]")
{
    auto t0 = rand<float>({10, 20, 8});
    auto chunks = chunk<4>(t0, /*dim=*/1);

    for (auto& chunk : chunks) {
        REQUIRE(chunk.size(0) == 10);
        REQUIRE(chunk.size(1) == 5);
        REQUIRE(chunk.size(2) == 8);
    }
}


TEST_CASE("Tensor chunk not divisible", "[chunk]")
{
    auto t0 = rand<float>({10, 20, 8});

    REQUIRE_THROWS_MATCHES(
        chunk<3>(t0, /*dim=*/2), std::invalid_argument,
        Catch::Matchers::Message("chunk: the tensor dimension (2) of size (8) is not "
                                 "divisible by a number of chunks 3")
    );
}

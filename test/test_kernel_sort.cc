// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/kernel/sort.h>
#include <metalchat/tensor/format.h>


using namespace metalchat;


TEST_CASE("Test bucketize", "[kernel::bucketize]")
{
    hardware_accelerator gpu0;
    kernel::bucketize<float> bucketize(gpu0);

    auto input = tensor<float, 2>({{3, 6, 9}, {3, 6, 9}});
    auto boundaries = tensor<float, 1>({1, 3, 5, 7, 9});

    auto output_right = bucketize(input, boundaries, /*right=*/true).get();
    auto expect_right = tensor<float, 2>({{2, 3, 5}, {2, 3, 5}});

    REQUIRE(output_right.dim() == 2);
    for (std::size_t i = 0; i < output_right.size(0); i++) {
        for (std::size_t j = 0; j < output_right.size(1); j++) {
            REQUIRE((output_right[i, j]) == (expect_right[i, j]));
        }
    }

    auto output_left = bucketize(input, boundaries, /*right=*/false).get();
    auto expect_left = tensor<float, 2>({{1, 3, 4}, {1, 3, 4}});

    REQUIRE(output_left.dim() == 2);
    for (std::size_t i = 0; i < output_left.size(0); i++) {
        for (std::size_t j = 0; j < output_left.size(1); j++) {
            REQUIRE((output_left[i, j]) == (expect_left[i, j]));
        }
    }
}


TEST_CASE("Test sorting", "[kernel::sort]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::sort<float> sort(gpu0);

    auto input = shared_tensor(rand<float>({1, 1, 100000}));
    auto [values_future, indices_future] = sort(input);

    auto values = values_future.get();
    auto indices = indices_future.get();

    REQUIRE(values.dim() == 3);
    REQUIRE(values.size(0) == input.size(0));
    REQUIRE(values.size(1) == input.size(1));
    REQUIRE(values.size(2) == input.size(2));

    REQUIRE(indices.dim() == 3);
    REQUIRE(indices.size(0) == input.size(0));
    REQUIRE(indices.size(1) == input.size(1));
    REQUIRE(indices.size(2) == input.size(2));

    for (std::size_t i = 0; i < values.size(0); i++) {
        for (std::size_t j = 0; j < values.size(1); j++) {
            auto values_ij = values[i][j];
            REQUIRE(std::is_sorted(values_ij.begin(), values_ij.end(), std::greater<float>()));

            auto indices_ij = indices[i][j];
            std::vector<float> values_out(indices_ij.size(0));

            for (std::size_t k = 0; k < values_out.size(); k++) {
                values_out[k] = input[i][0][values_ij[k]];
            }
            REQUIRE(std::is_sorted(values_out.begin(), values_out.end(), std::greater<float>()));
        }
    }
}


TEST_CASE("Sorting benchmark", "[!benchmark][kernel::sort]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::sort<float> sort(gpu0);

    auto input_cpu = rand<float>({1, 1, 128256});
    auto input = shared_tensor(empty<float>({1, 1, 128256}, gpu0.get_allocator()));
    std::copy(input_cpu.begin(), input_cpu.end(), input.begin());

    BENCHMARK("sort 128256 elements")
    {
        auto [values_future, indices_future] = sort(input);
        return values_future.get();
    };
}

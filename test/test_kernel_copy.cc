// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/kernel/copy.h>


using namespace metalchat;


TEST_CASE("Copy 2-dimensional tensors", "[kernel::clone]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::clone<float> clone(gpu0);

    auto input = shared_tensor(rand<float>({16, 64}));
    auto output = shared_tensor(empty<float>({16, 64}, gpu0.get_allocator()));

    clone(input, output).wait();

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            REQUIRE((input[i, j]) == (output[i, j]));
        }
    }
}


TEST_CASE("Copy large 2-dimensional tensors", "[kernel::clone]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::clone<float> clone(gpu0);

    auto input = shared_tensor(rand<float>({16, 4096}));
    auto output = shared_tensor(empty<float>({16, 4096}, gpu0.get_allocator()));

    clone(input, output).wait();

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            REQUIRE((input[i, j]) == (output[i, j]));
        }
    }
}


TEST_CASE("Copy into slice", "[kernel::clone]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::clone<float> clone(gpu0);

    auto input = shared_tensor(rand<float>({1, 6, 8, 1, 64}));
    auto output = shared_tensor(full<float>({1, 6, 8, 4, 64}, 0.0, gpu0.get_allocator()));

    auto target = output.narrow(/*dim=*/3, /*offset=*/2, /*length=*/1);
    clone(input.view({-1, 64}), target.view({-1, 64})).wait();

    for (std::size_t i0 = 0; i0 < input.size(0); i0++) {
        for (std::size_t i1 = 0; i1 < input.size(1); i1++) {
            for (std::size_t i2 = 0; i2 < input.size(1); i2++) {
                for (std::size_t i4 = 0; i4 < input.size(1); i4++) {
                    REQUIRE((input[i0, i1, i2, 0, i4]) == (output[i0, i1, i2, 2, i4]));
                }
            }
        }
    }
}


TEST_CASE("Inplace scalar scatter", "[kernel::scatter]")
{
    hardware_accelerator gpu0;
    kernel::scatter<float> scatter(gpu0);

    auto input = shared_tensor(full<float>({16, 128}, 1.0f, gpu0.get_allocator()));
    auto mask = shared_tensor(rand<bool>({16, 128}));

    auto output = scatter(input, mask, 9.0f).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == input.size(0));
    REQUIRE(output.size(1) == input.size(1));

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            if (mask[i, j]) {
                REQUIRE_THAT((input[i, j]), Catch::Matchers::WithinAbs(9.0f, 0.0001));
            } else {
                REQUIRE_THAT((input[i, j]), Catch::Matchers::WithinAbs(1.0f, 0.0001));
            }
        }
    }
}


TEST_CASE("Inplace tensor scatter", "[kernel::scatter]")
{
    hardware_accelerator gpu0;
    kernel::scatter<float> scatter(gpu0);

    auto output = shared_tensor(full<float>({16, 128}, 1.0f, gpu0.get_allocator()));
    auto index = shared_tensor(rand<int32_t>({16, 8}, 0, 127));
    auto input = shared_tensor(rand<float>({16, 8}));

    for (std::size_t i = 0; i < index.size(0); i++) {
        std::unordered_set<int32_t> set;
        for (std::size_t j = 0; j < index.size(1); j++) {
            while (set.contains(index[i, j])) {
                index[i, j] = (index[i, j] + 1) % 128;
            }
            set.insert(index[i, j]);
        }
    }

    auto result = scatter(output, index, input).get();
    REQUIRE(result.dim() == 2);
    REQUIRE(result.size(0) == output.size(0));
    REQUIRE(result.size(1) == output.size(1));

    for (std::size_t i = 0; i < index.size(0); i++) {
        for (std::size_t j = 0; j < index.size(1); j++) {
            REQUIRE((result[i, index[i, j]]) == (input[i, j]));
        }
    }
}


TEST_CASE("Gather by index", "[kernel::gather]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::gather<float> gather(gpu0);

    auto input = shared_tensor(rand<float>({16, 128}));
    auto index = shared_tensor(rand<int32_t>({16, 10}, 0, 127));

    auto output = gather(input, index).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == index.size(0));
    REQUIRE(output.size(1) == index.size(1));

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            REQUIRE((output[i, j]) == (input[i, index[i, j]]));
        }
    }
}


TEST_CASE("Conditional copy", "[kernel::where]")
{
    hardware_accelerator gpu0;
    kernel::where<float> where(gpu0);

    auto cond = rand<bool>({8, 16});
    auto input = rand<float>({8, 16});
    auto other = rand<float>({8, 16});

    auto output = where(cond, input, other).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == cond.size(0));
    REQUIRE(output.size(1) == cond.size(1));

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            const auto& expect = cond[i, j] ? input[i, j] : other[i, j];
            REQUIRE((output[i, j]) == expect);
        }
    }
}

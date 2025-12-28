// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <iterator>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Iterate 3d tensor slice", "[tensor_iterator]")
{
    auto t = rand<float>({10, 7, 6});
    auto s = t[slice(2, 7), slice(4, 7), slice(2, 4)];

    std::vector<float> data;

    auto last = s.end();
    for (auto first = s.begin(); first != last; ++first) {
        data.push_back(*first);
        REQUIRE(data.size() <= s.numel()); // Too many iterations.
    }

    REQUIRE(data.size() == s.numel());

    auto first = data.begin();

    for (std::size_t i = 0; i < s.size(0); i++) {
        for (std::size_t j = 0; j < s.size(1); j++) {
            for (std::size_t k = 0; k < s.size(2); k++) {
                REQUIRE(s[i, j, k] == (*first));
                ++first;
            }
        }
    }
}


TEST_CASE("Iterator outside numel", "[tensor_iterator]")
{
    auto t = rand<float>({3, 4});
    auto it0 = tensor_iterator(t, 30);
    REQUIRE(it0 == t.end());

    std::advance(it0, 10);
    REQUIRE(it0 == t.end());

    auto it1 = tensor_iterator(t, 1);
    it1 = it1 + 10;
    REQUIRE((*tensor_iterator(t, 11)) == (*it1));
}


TEST_CASE("Iterator of different views", "[tensor_iterator]")
{
    auto t = rand<float>({4, 4, 4});
    auto v = t.view({1, 16, 4});

    REQUIRE(t.begin() == v.begin());
    REQUIRE(t.end() == v.end());

    auto s = t[slice(0, 2), slice(0, 2), slice(0, 2)];
    REQUIRE(t.begin() != s.begin());
    REQUIRE(t.end() != s.end());
}


TEST_CASE("Iterator of sub-tensors", "[tensor_iterator]")
{
    auto t = rand<float>({1, 1, 200});
    auto s = t[0][0];

    REQUIRE(std::equal(s.begin(), s.end(), s.data_ptr()));
}

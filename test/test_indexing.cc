#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


TEST_CASE("Read-write 2d tensor slicing", "[tensor::operator]")
{
    auto t = full<bf16>({4, 5}, 5.0);
    for (auto i = 0; i < t.size(0); i++) {
        for (auto j = 0; j < t.size(1); j++) {
            REQUIRE(t[i][j] == 5.0);
            t[i][j] = bf16((i + 1) * 10 + j);
        }
    }

    auto s = t[slice(1, 3), slice(1, 4)];
    REQUIRE(s.size(0) == 2);
    REQUIRE(s.size(1) == 3);

    REQUIRE(s.stride(0) == 5);
    REQUIRE(s.stride(1) == 1);
    REQUIRE(!s.is_contiguous());

    for (auto i = 0; i < s.size(0); i++) {
        for (auto j = 0; j < s.size(1); j++) {
            s[i][j] = 0.0;
        }
    }

    // Ensure that writing through the tensor view updates values in the
    // underlying storage.
    for (auto i = 1; i < 3; i++) {
        for (auto j = 1; j < 4; j++) {
            REQUIRE(t[i][j] == 0.0);
        }
    }
}


TEST_CASE("Read-write 1d tensor slicing", "[tensor::operator]")
{
    auto t = full<bf16>({15}, 2.0);
    auto s = t[slice(3, 10)];

    REQUIRE(s.size(0) == 7);
    REQUIRE(s.stride(0) == 1);
    REQUIRE(!s.is_contiguous());

    for (auto i = 0; i < s.size(0); i++) {
        s[i] = 0.0;
    }

    for (auto i = 3; i < 10; i++) {
        REQUIRE(t[i] == 0.0);
    }
}


TEST_CASE("Copy 2d tensor through slicing", "[tensor::operator=]")
{
    auto t0 = full<bf16>({7, 8}, 2.0);
    auto t1 = rand<bf16>({3, 2});

    t0[slice(4, 7), slice(6, 8)] = t1;

    REQUIRE(t0.size(0) == 7);
    REQUIRE(t0.size(1) == 8);

    for (std::size_t i = 4; i < 7; i++) {
        for (std::size_t j = 6; j < 8; j++) {
            REQUIRE(t0[i][j] == t1[i - 4][j - 6]);
        }
    }
}

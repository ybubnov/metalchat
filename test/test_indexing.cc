#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


TEST_CASE("Read-write tensor slicing", "[tensor::operator]")
{
    auto T = full<bf16>({4, 5}, 5.0);
    for (auto i = 0; i < T.size(0); i++) {
        for (auto j = 0; j < T.size(1); j++) {
            REQUIRE(T[i][j] == 5.0);
            T[i][j] = bf16((i + 1) * 10 + j);
        }
    }

    auto S = T[slice(1, 3), slice(1, 4)];
    REQUIRE(S.size(0) == 2);
    REQUIRE(S.size(1) == 3);

    REQUIRE(S.stride(0) == 5);
    REQUIRE(S.stride(1) == 1);
    REQUIRE(!S.is_contiguous());

    for (auto i = 0; i < S.size(0); i++) {
        for (auto j = 0; j < S.size(1); j++) {
            S[i][j] = 0.0;
        }
    }

    // Ensure that writing through the tensor view updates values in the
    // underlying storage.
    for (auto i = 1; i < 3; i++) {
        for (auto j = 1; j < 4; j++) {
            REQUIRE(T[i][j] == 0.0);
        }
    }
}

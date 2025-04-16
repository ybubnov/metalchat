#include <catch2/catch_test_macros.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::indexing;


TEST_CASE("Tensor transpose", "[tensor::transpose]")
{
    auto x = rand<float>({2, 3, 4});
    auto x_t = x.transpose(0, 2, 1);

    REQUIRE(x_t.size(0) == 2);
    REQUIRE(x_t.size(1) == 4);
    REQUIRE(x_t.size(2) == 3);

    x[1][2][3] = 10.0;
    REQUIRE(x_t[1][3][2] == 10.0);
}


TEST_CASE("Tensor slice transpose", "[tensor::transpose]")
{
    auto x = rand<float>({5, 4, 3, 2});
    auto y = x[slice(0, 1), slice(1, 3), slice(0, 2), slice(1, 2)];
    REQUIRE(y.size(0) == 1);
    REQUIRE(y.size(1) == 2);
    REQUIRE(y.size(2) == 2);
    REQUIRE(y.size(3) == 1);

    auto y_t = y.transpose(1, 0, 3, 2);
    REQUIRE(y_t.size(0) == 2);
    REQUIRE(y_t.size(1) == 1);
    REQUIRE(y_t.size(2) == 1);
    REQUIRE(y_t.size(3) == 2);

    std::fill(y_t.begin(), y_t.end(), 0.0);

    for (auto i = 0; i < 1; i++) {
        for (auto j = 1; j < 3; j++) {
            for (auto k = 0; k < 2; k++) {
                for (auto l = 1; l < 2; l++) {
                    REQUIRE(x[i][j][k][l] == 0.0);
                }
            }
        }
    }
}


TEST_CASE("Tensor transpose in scope", "[tensor::transpose]")
{
    auto x = []() {
        auto x = full<float>({3, 4, 2, 2}, 7.0);
        return x.transpose(0, 2, 3, 1);
    }();

    REQUIRE(x.dim() == 4);
    REQUIRE(x.size(0) == 3);
    REQUIRE(x.size(1) == 2);
    REQUIRE(x.size(2) == 2);
    REQUIRE(x.size(3) == 4);

    for (const auto& v : x) {
        REQUIRE(v == 7.0);
    }
}

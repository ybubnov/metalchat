#include <ranges>

#include <catch2/catch_test_macros.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor concatenate", "[concatenate]")
{
    auto t0 = full<float>({3, 4, 2}, 1.0);
    auto t1 = full<float>({3, 4, 2}, 2.0);
    auto t2 = full<float>({3, 4, 2}, 3.0);
    auto t3 = full<float>({3, 4, 2}, 4.0);
    auto t4 = full<float>({3, 4, 2}, 5.0);

    auto tensors = {std::cref(t0), std::cref(t1), std::cref(t2), std::cref(t3), std::cref(t4)};

    auto output0 = concatenate(tensors, 0);
    REQUIRE(output0.dim() == 3);
    REQUIRE(output0.size(0) == 15);
    REQUIRE(output0.size(1) == 4);
    REQUIRE(output0.size(2) == 2);
    REQUIRE(output0[0][0][0] == 1.0);
    REQUIRE(output0[14][3][1] == 5.0);

    auto output1 = concatenate(tensors, 1);
    REQUIRE(output1.dim() == 3);
    REQUIRE(output1.size(0) == 3);
    REQUIRE(output1.size(1) == 20);
    REQUIRE(output1.size(2) == 2);
    REQUIRE(output1[0][0][0] == 1.0);
    REQUIRE(output1[2][19][1] == 5.0);

    auto output2 = concatenate(tensors, 2);
    REQUIRE(output2.dim() == 3);
    REQUIRE(output2.size(0) == 3);
    REQUIRE(output2.size(1) == 4);
    REQUIRE(output2.size(2) == 10);
    REQUIRE(output2[0][0][0] == 1.0);
    REQUIRE(output2[2][3][9] == 5.0);
}

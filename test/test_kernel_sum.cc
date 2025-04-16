#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Sum of 3-dimensional tensors", "[kernel::sum]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sum<float> sum(gpu0);

    auto input1 = shared_tensor(rand<float>({1, 4, 2048}));
    auto input2 = shared_tensor(rand<float>({1, 4, 2048}));
    auto output = sum(input1, input2).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 2048);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                REQUIRE_THAT(
                    (output[i, j, k]),
                    Catch::Matchers::WithinAbs(input1[i, j, k] + input2[i, j, k], 0.00001)
                );
            }
        }
    }
}


TEST_CASE("2-dimensional sum of tensors", "[kernel::sum2]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sum2<float> sum(gpu0);

    auto input1 = shared_tensor(rand<float>({5, 32, 16, 16}));
    auto input2 = shared_tensor(rand<float>({16, 16}));
    auto output = sum(input1, input2).get();

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 5);
    REQUIRE(output.size(1) == 32);
    REQUIRE(output.size(2) == 16);
    REQUIRE(output.size(3) == 16);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto m = 0; m < output.size(2); m++) {
                for (auto n = 0; n < output.size(3); n++) {
                    REQUIRE_THAT(
                        (output[i, j, m, n]),
                        Catch::Matchers::WithinAbs(input1[i, j, m, n] + input2[m, n], 0.00001)
                    );
                }
            }
        }
    }
}

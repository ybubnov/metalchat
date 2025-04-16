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


TEST_CASE("Sum of 4-dimensional tensors", "[kernel::sum]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sum<float> sum(gpu0);

    auto input1 = rand<float>({1, 4, 2048});
    auto input2 = rand<float>({1, 4, 2048});
    auto output = sum(input1, input2);

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

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional/sum.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Sum of 4-dimensional tensors", "[functional::sum]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sum<bf16> sum(gpu0);

    auto input1 = rand<bf16>({4, 4, 4, 128});
    auto input2 = rand<bf16>({4, 4, 4, 128});
    auto output = sum(input1, input2);

    REQUIRE(output.dim() == 4);
    REQUIRE(output.numel() == input1.numel());
    REQUIRE(output.numel() == input2.numel());

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                for (auto l = 0; l < output.size(3); l++) {
                    REQUIRE_THAT(
                        output[i][j][k][l],
                        Catch::Matchers::WithinAbs(input1[i][j][k][l] + input2[i][j][k][l], 0.01)
                    );
                }
            }
        }
    }
}

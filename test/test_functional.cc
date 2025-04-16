#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Test repeat interleave", "[functional::repeat_interleave]")
{
    auto original = rand<float>({1, 8, 7, 64});
    auto t = empty<float>({1, 8, 7, 64});
    t = original;

    auto output = repeat_interleave(std::move(original), 4, /*dim=*/1);

    REQUIRE(output.dim() == 5);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 8);
    REQUIRE(output.size(2) == 4);
    REQUIRE(output.size(3) == 7);
    REQUIRE(output.size(4) == 64);

    for (auto i = 0; i < t.size(0); i++) {
        for (auto j = 0; j < t.size(1); j++) {
            for (auto k = 0; k < t.size(2); k++) {
                for (auto l = 0; l < t.size(3); l++) {
                    for (auto m = 0; m < output.size(2); m++) {
                        REQUIRE(t[i, j, k, l] == output[i, j, m, k, l]);
                    }
                }
            }
        }
    }
}

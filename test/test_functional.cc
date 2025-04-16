#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Test repeat interleave", "[functional::repeat_interleave]")
{
    auto original = rand<float>({1, 6, 8, 64});
    auto t = empty<float>({1, 6, 8, 64});
    t = original;

    metalchat::device gpu0("metalchat.metallib");
    metalchat::cpy<float> cp(gpu0);

    auto output = repeat_interleave(std::move(original), 4, /*dim=*/2, cp, gpu0);

    REQUIRE(output.dim() == 5);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 6);
    REQUIRE(output.size(2) == 8);
    REQUIRE(output.size(3) == 4);
    REQUIRE(output.size(4) == 64);

    std::cout << output << std::endl;

    for (auto i = 0; i < t.size(0); i++) {
        for (auto j = 0; j < t.size(1); j++) {
            for (auto k = 0; k < t.size(2); k++) {
                for (auto l = 0; l < t.size(3); l++) {
                    for (auto m = 0; m < output.size(3); m++) {
                        REQUIRE(t[i, j, k, l] == output[i, j, k, m, l]);
                    }
                }
            }
        }
    }
}

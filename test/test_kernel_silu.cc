#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/silu.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("SiLU function", "[kernel::hadamard]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::silu<float> silu(gpu0);

    auto input = rand<float>({5, 8192});

    auto output = silu(input);
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 5);
    REQUIRE(output.size(1) == 8192);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            auto result = input[i, j] / (1 + std::exp(-input[i, j]));
            REQUIRE_THAT((output[i, j]), Catch::Matchers::WithinAbs(result, 0.00001));
        }
    }
}

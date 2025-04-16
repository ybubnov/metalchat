#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Hadamard product", "[kernel::hadamard]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::hadamard<float> m(gpu0);

    auto input1 = rand<float>({5, 8192});
    auto input2 = rand<float>({5, 8192});

    auto output = m(input1, input2);
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 5);
    REQUIRE(output.size(1) == 8192);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            auto result = input1[i, j] * input2[i, j];
            REQUIRE_THAT((output[i, j]), Catch::Matchers::WithinAbs(result, 0.00001));
        }
    }
}

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/arithmetic.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Add 3-dimensional tensors", "[kernel::add]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::add<float> add(gpu0);

    auto input1 = shared_tensor(rand<float>({1, 4, 2048}));
    auto input2 = shared_tensor(rand<float>({1, 4, 2048}));
    auto output = add(input1, input2).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 2048);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                REQUIRE_THAT(
                    (output[i, j, k]),
                    Catch::Matchers::WithinAbs(input1[i, j, k] + input2[i, j, k], 0.00001)
                );
            }
        }
    }
}


TEST_CASE("Sub 3-dimensional tensors", "[kernel::sub]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::sub<float> sub(gpu0);

    auto input1 = shared_tensor(rand<float>({1, 4, 2048}));
    auto input2 = shared_tensor(rand<float>({1, 4, 2048}));
    auto output = sub(input1, input2).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 2048);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                REQUIRE_THAT(
                    (output[i, j, k]),
                    Catch::Matchers::WithinAbs(input1[i, j, k] - input2[i, j, k], 0.00001)
                );
            }
        }
    }
}


TEST_CASE("Add 2-dimensional tensors", "[kernel::add2]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::add2<float> add(gpu0);

    auto input1 = shared_tensor(rand<float>({5, 32, 16, 16}));
    auto input2 = shared_tensor(rand<float>({16, 16}));
    auto output = add(input1, input2).get();

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 5);
    REQUIRE(output.size(1) == 32);
    REQUIRE(output.size(2) == 16);
    REQUIRE(output.size(3) == 16);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t m = 0; m < output.size(2); m++) {
                for (std::size_t n = 0; n < output.size(3); n++) {
                    REQUIRE_THAT(
                        (output[i, j, m, n]),
                        Catch::Matchers::WithinAbs(input1[i, j, m, n] + input2[m, n], 0.00001)
                    );
                }
            }
        }
    }
}

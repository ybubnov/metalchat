#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


/*
TEST_CASE("Matmul 4d predefined", "[kernel::bmm]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::bmm<bf16> mm(gpu0);

    auto input1 = rand<bf16>({3, 3, 3, 5});
    auto input2 = rand<bf16>({3, 3, 5, 7});
    auto output = mm(input1, input2);

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 3);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 7);
}
*/


TEST_CASE("Matmul single batch multiplication", "[kernel::bmm]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::bmm<float> mm(gpu0);

    auto input1 = rand<float>({1, 5, 2048});     // b, i, j
    auto input2 = rand<float>({8192, 2048}).t(); // j, k
    // auto input2 = rand<float>({2048, 8192}); // j, k

    auto output = mm(input1, input2);

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 5);
    REQUIRE(output.size(2) == 8192);

    for (auto batch = 0; batch < input1.size(0); batch++) {
        for (auto i = 0; i < input1.size(1); i++) {
            for (auto k = 0; k < input2.size(1); k++) {
                float result_ik = 0;
                for (auto j = 0; j < input1.size(2); j++) {
                    result_ik += (input1[batch, i, j] * input2[j, k]);
                }

                REQUIRE_THAT((output[batch, i, k]), Catch::Matchers::WithinAbs(result_ik, 0.0001));
            }
        }
    }
}


TEST_CASE("Matmul large 2d", "[kernel::bmm]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::bmm<float> mm(gpu0);

    auto input1 = full<float>({8, 2048}, 2.0);
    auto input2 = full<float>({2048, 128256}, 1.0);
    auto output = mm(input1, input2);

    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 8);
    REQUIRE(output.size(1) == 128256);

    std::cout << output << std::endl;
}

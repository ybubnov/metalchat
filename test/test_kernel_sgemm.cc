#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/sgemm.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Matmul 4d predefined", "[functional::sgemm]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sgemm<bf16> mm(gpu0);

    auto input1 = rand<bf16>({3, 3, 3, 5});
    auto input2 = rand<bf16>({3, 3, 5, 7});
    auto output = mm(input1, input2);

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 3);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 7);
}
